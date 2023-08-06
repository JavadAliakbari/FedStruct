import logging
from ast import List
from copy import deepcopy
from itertools import product
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset, Planetoid
import numpy as np
from torch.nn.functional import normalize
from torch_geometric.nn import MessagePassing
from tqdm import tqdm
from src.GNN_classifier import GNNClassifier

from src.utils import config
from src.utils.graph import Graph
from src.utils.graph_partinioning import louvain_graph_cut
from src.models.GNN_models import GNN, MLP, calc_accuracy


class JointModel(torch.nn.Module):
    """Joint Model"""

    def __init__(
        self,
        clients,
        structure_layer_sizes,
        num_classes=None,
        client_layer_sizes=None,
        dropout=config.dropout,
        last_layer="softmax",
        logger=None,
    ):
        super().__init__()
        self.LOGGER = logger or logging

        self.num_layers = len(structure_layer_sizes) - 1
        self.dropout = dropout
        self.last_layer = last_layer
        self.models = nn.ParameterDict()
        if isinstance(clients, int):
            self.num_clients = num_clients

            for id in range(self.num_clients):
                model = MLP(
                    layer_sizes=client_layer_sizes,
                    dropout=dropout,
                    softmax=False,
                )

                self.models[f"client{id}"] = model

        elif isinstance(clients, list):
            self.num_clients = len(clients)

            for id in range(self.num_clients):
                model = clients[id]
                self.models[f"client{id}"] = model

        self.models[f"structure_model"] = GNN(
            in_dims=structure_layer_sizes,
            dropout=dropout,
            last_layer="linear",
        )

        if self.num_clients > 0:
            self.linear = nn.Linear(
                self.models[f"client{id}"][-1].out_features + structure_layer_sizes[-1],
                num_classes,
            )

        self.optimizers = self.create_optimizers()
        self.criterion = torch.nn.CrossEntropyLoss()

        self.message_passing = MessagePassing(aggr="mean")

    def __getitem__(self, item):
        return self.models[item]

    def create_optimizers(self):
        optimizers = {}
        for model_name, layers in self.models.items():
            # if not model_name.startswith("client"):
            #     continue
            parameters = list(layers.parameters())
            # parameters += list(self.models[f"structure_model"].parameters())
            optimizer = torch.optim.Adam(parameters, lr=config.lr, weight_decay=5e-4)
            optimizers[model_name] = optimizer

        return optimizers

    def reset_parameters(self) -> None:
        for model in self.models.values():
            model.reset_parameters()

    def reset_optimizers(self) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

    def state_dict(self):
        model_weights = {}
        for model_name, model in self.models.items():
            layer_weights = model.state_dict()
            model_weights[model_name] = layer_weights

        return model_weights

    def load_state_dict(self, model_weights: dict) -> None:
        for model_name, model in self.models.items():
            model.load_state_dict(model_weights[model_name])

    def forward(self, subgraphs: List, structure_graph: Graph):
        assert len(subgraphs) <= self.num_clients, "There is not any models as data"

        num_subgraphs = len(subgraphs)

        H = deque()
        for client_id in range(num_subgraphs):
            x = subgraphs[client_id].x
            h = self.models[f"client{client_id}"](x)
            H.append(h)

        x = structure_graph.structural_features
        edge_index = structure_graph.edge_index
        model = self.models["structure_model"]
        S = model(x, edge_index)

        for client_id in range(num_subgraphs):
            node_ids = subgraphs[client_id].node_ids
            x_s = S[node_ids].clone()
            x = torch.hstack((H.popleft(), x_s))
            h = self.linear(x)
            H.append(h)

        out = {}
        out[f"structure_model"] = S
        for client_id in range(num_subgraphs):
            x = H.popleft()
            if self.last_layer == "softmax":
                out[f"client{client_id}"] = F.softmax(x, dim=1)
            elif self.last_layer == "relu":
                out[f"client{client_id}"] = F.relu(x)
            else:
                out[f"client{client_id}"] = x

        return out

    def cosine_similarity(h1, h2):
        return torch.dot(h1, h2) / (
            (torch.norm(h1) + 0.000001) * (torch.norm(h2) + 0.000001)
        )

    def calc_loss(self, embeddings, graph, negative_samples_ratio=0.2):
        loss = torch.zeros(embeddings.shape[0], dtype=torch.float32)
        node_ids = graph.node_ids.numpy()

        normalized_embeddings = normalize(embeddings, dim=1)

        edge_index = graph.edge_index
        neighbors_embeddings = self.message_passing.propagate(
            edge_index, x=normalized_embeddings
        )

        negative_edge_index = []
        negative_samples_list = []
        for node_id in node_ids:
            other_nodes = graph.negative_samples[node_id]

            neighbor_size = graph.degree[node_id].item()
            negative_size = max(5, neighbor_size)
            negative_samples = np.random.choice(other_nodes, negative_size)
            negative_samples_list.append(negative_samples)
            negative_edge_index += list(product(negative_samples, [node_id]))

        negative_edge_index = torch.tensor(np.array(negative_edge_index).T)
        negative_embeddings = self.message_passing.propagate(
            negative_edge_index, x=normalized_embeddings
        )

        diff = negative_embeddings - neighbors_embeddings

        loss = (1 + torch.einsum("ij,ij->i", diff, normalized_embeddings)) / 2

        return loss

    def train_model(self, graph: Graph, subgraphs=[]):
        self.train()
        out = self.forward(subgraphs, graph)
        loss = self.calc_loss(out[f"structure_model"], graph)

        return out, loss

    def fit(self, graph, clients, epochs=1):
        metrics = {}
        subgraphs = []
        plot_results = {}
        for client in clients:
            subgraphs.append(client.subgraph)
            plot_results[f"Client{client.id}"] = []

        bar = tqdm(total=epochs, position=0)
        for epoch in range(epochs):
            self.reset_optimizers()
            out, structure_loss = self.train_model(graph, subgraphs)
            loss_list = torch.zeros(len(subgraphs), dtype=torch.float32)
            total_loss = 0
            for ind, client in enumerate(clients):
                node_ids = client.get_nodes()
                y = client.subgraph.y
                train_mask = client.subgraph.train_mask
                val_mask = client.subgraph.val_mask

                client_structure_loss = structure_loss[node_ids].clone()
                y_pred = out[f"client{client.id}"].clone()

                cls_train_loss = self.criterion(y_pred[train_mask], y[train_mask])
                str_train_loss = client_structure_loss[train_mask].mean()
                train_loss = cls_train_loss + str_train_loss
                loss_list[ind] = train_loss

                train_acc = calc_accuracy(
                    y_pred[train_mask].argmax(dim=1),
                    y[train_mask],
                )

                # Validation
                self.eval()
                with torch.no_grad():
                    if val_mask.any():
                        cls_val_loss = self.criterion(y_pred[val_mask], y[val_mask])
                        str_val_loss = client_structure_loss[val_mask].mean()
                        val_loss = cls_val_loss + str_val_loss

                        val_acc = calc_accuracy(
                            y_pred[val_mask].argmax(dim=1), y[val_mask]
                        )
                    else:
                        cls_val_loss = 0
                        val_acc = 0

                    result = {
                        "Train Loss": round(cls_train_loss.item(), 4),
                        "Train Acc": round(train_acc, 4),
                        "Val Loss": round(cls_val_loss.item(), 4),
                        "Val Acc": round(val_acc, 4),
                        "Total Train Loss": round(train_loss.item(), 4),
                        "Total Val Loss": round(val_loss.item(), 4),
                        "Epoch": epoch,
                    }

                    plot_results[f"Client{client.id}"].append(result)
                    metrics[f"client{client.id}"] = result["Val Acc"]

                if epoch == epochs - 1:
                    self.LOGGER.info(f"JTSW results for client{client.id}:")
                    self.LOGGER.info(f"{result}")

            total_loss = loss_list.mean()
            total_loss.backward()

            for optimizer in self.optimizers.values():
                optimizer.step()

            with torch.no_grad():
                metrics["Total Loss"] = round(total_loss.item(), 4)
                metrics["Structure Loss"] = round(structure_loss.mean().item(), 4)

                model_weights = self.state_dict()
                sum_weights = None
                for client in clients:
                    sum_weights = JointModel.add_weights(
                        sum_weights, model_weights[f"client{client.id}"]
                    )

                mean_weights = JointModel.calc_mean_weights(sum_weights, len(clients))
                for client in clients:
                    model_weights[f"client{id}"] = mean_weights

                self.load_state_dict(model_weights)

            bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
            bar.set_postfix(metrics)
            bar.update()

        test_metrics = self.test(clients, graph)
        for client in clients:
            test_acc = test_metrics[f"Client{client.id}"]["Test Acc"]
            self.LOGGER.info(f"Client{client.id} test accuracy: {test_acc}")

            GNNClassifier.plot_results(
                plot_results[f"Client{client.id}"],
                client.id,
                type="JTSW",
            )

    @torch.no_grad()
    def add_weights(sum_weights, weights):
        if sum_weights is None:
            sum_weights = deepcopy(weights)
        else:
            for layer_name, layer_parameters in weights.items():
                for component_name, component_parameters in layer_parameters.items():
                    sum_weights[layer_name][component_name] = (
                        sum_weights[layer_name][component_name] + component_parameters
                    )
        return sum_weights

    @torch.no_grad()
    def calc_mean_weights(sum_weights, count):
        for layer_name, layer_parameters in sum_weights.items():
            for component_name, component_parameters in layer_parameters.items():
                sum_weights[layer_name][component_name] = component_parameters / count

        return sum_weights

    @torch.no_grad()
    def test(self, clients, graph):
        self.eval()
        metrics = {}
        subgraphs = []
        for client in clients:
            subgraphs.append(client.subgraph)
        out = self.forward(subgraphs, graph)

        for client in clients:
            y = client.subgraph.y
            test_mask = client.subgraph.test_mask

            y_pred = out[f"client{client.id}"]

            test_acc = calc_accuracy(
                y_pred[test_mask].argmax(dim=1),
                y[test_mask],
            )

            result = {
                "Test Acc": round(test_acc, 4),
            }

            metrics[f"Client{client.id}"] = result

        return metrics


if __name__ == "__main__":
    dataset = Planetoid(root="/tmp/Cora", name="Cora")

    node_ids = torch.arange(dataset[0].num_nodes)
    graph = Graph(
        x=dataset[0].x,
        y=dataset[0].y,
        edge_index=dataset[0].edge_index,
        node_ids=node_ids,
    )

    graph.add_masks(train_size=0.5, test_size=0.2)
    graph.add_structural_features(config.num_structural_features)

    num_classes = dataset.num_classes

    subgraphs = louvain_graph_cut(graph)
    num_clients = len(subgraphs)

    GNN = JointModel(
        num_clients=num_clients,
        client_layer_sizes=[graph.num_features] + config.classifier_layer_sizes,
        structure_layer_sizes=[config.num_structural_features]
        + config.classifier_layer_sizes,
        num_classes=num_classes,
    )

    criterion = torch.nn.CrossEntropyLoss()
    parameters = list(GNN.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.lr, weight_decay=5e-4)

    GNN.forward(subgraphs, graph)
    a = 2
