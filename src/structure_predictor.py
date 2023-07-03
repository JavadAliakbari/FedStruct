from ast import List
from itertools import compress

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils import config
from src.utils.graph import Graph
from src.models.structure_model import JointModel
from src.models.GNN_models import GraphSAGE, calc_accuracy, test


class StructurePredictor:
    def __init__(self, id, edge_index, node_ids):
        self.id = id
        self.edge_index = edge_index
        self.node_ids = node_ids

    def prepare_data(self):
        self.graph = Graph(
            edge_index=self.edge_index,
            node_ids=self.node_ids,
        )

        self.graph.add_structural_features()
        self.graph.add_masks()

    def set_model(self, num_clients, num_features, num_classes):
        client_layer_sizes = [num_features] + config.classifier_layer_sizes
        structure_layer_sizes = [
            self.graph.num_structural_features
        ] + config.structure_layers_size

        self.model = JointModel(
            num_clients=num_clients,
            client_layer_sizes=client_layer_sizes,
            structure_layer_sizes=structure_layer_sizes,
            num_classes=num_classes,
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr, weight_decay=5e-4
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def initialize_structural_graph(self, num_clients, num_features, num_classes):
        self.prepare_data()
        self.set_model(num_clients, num_features, num_classes)

    def reset_parameters(self):
        self.model.reset_parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, weights):
        self.model.load_state_dict(weights)

    def cosine_similarity(h1, h2):
        return (
            torch.dot(h1, h2)
            / (torch.norm(h1) + 0.000001)
            / (torch.norm(h2) + 0.000001)
        )

    def find_negative_samples(node_ids, neighbors, negative_samples_ratio=0.2):
        neighbor_nodes_mask = torch.isin(node_ids, neighbors)
        other_nodes = node_ids[~neighbor_nodes_mask]
        negative_size = negative_samples_ratio * other_nodes.shape[0]
        negative_samples_masks = torch.multinomial(other_nodes, negative_size)
        negative_samples = other_nodes[negative_samples_masks]

        return negative_samples

    # def calc_loss(self, embeddings, negative_samples_ratio=0.2):
    #     loss = torch.zeros(embeddings.shape[0], dtype=torch.float32)
    #     node_ids = self.graph.node_ids.numpy()
    #     for node_index, node_id in enumerate(node_ids):
    #         node_mask = self.graph.node_ids == node_id
    #         neighbors = list(compress(self.graph.node_neighbors, node_mask))[0]

    #         neighbor_nodes_mask = np.isin(node_ids, neighbors)
    #         other_nodes = node_ids[~neighbor_nodes_mask]
    #         negative_size = len(neighbors)
    #         # negative_size = int(negative_samples_ratio * other_nodes.shape[0])
    #         negative_samples = np.random.choice(other_nodes, negative_size)
    #         negative_samples_mask = np.isin(node_ids, negative_samples)

    #         # negative_samples = StructurePredictor.find_negative_samples(self.graph.node_ids,neighbors,negative_samples_ratio=0.2,)

    #         h1 = embeddings[node_mask].squeeze(0)
    #         neighbors_embedding = embeddings[neighbor_nodes_mask]
    #         negative_samples_embedding = embeddings[negative_samples_mask]
    #         loss_list = torch.zeros(
    #             (len(neighbors) + negative_size), dtype=torch.float32
    #         )
    #         ind = 0
    #         for h2 in neighbors_embedding:
    #             neighbor_loss = 1 - StructurePredictor.cosine_similarity(h1, h2)
    #             loss_list[ind] += neighbor_loss
    #             ind += 1

    #         for h2 in negative_samples_embedding:
    #             negative_loss = StructurePredictor.cosine_similarity(h1, h2)
    #             loss_list[ind] += negative_loss
    #             ind += 1

    #         if loss_list.shape[0] > 0:
    #             node_loss = loss_list.mean()
    #         else:
    #             node_loss = torch.tensor([0], dtype=torch.float32)
    #         loss[node_index] = node_loss

    #     return loss

    def calc_loss(self, embeddings, negative_samples_ratio=0.2):
        loss = torch.zeros(embeddings.shape[0], dtype=torch.float32)
        node_ids = self.graph.node_ids.numpy()

        for node_index, node_id in enumerate(node_ids):
            neighbors = self.graph.node_neighbors[node_id]
            other_nodes = self.graph.negative_samples[node_id]

            negative_size = max(5, self.graph.degree[node_id].item())
            negative_samples = np.random.choice(other_nodes, negative_size)

            h1 = embeddings[node_id].squeeze(0)
            neighbors_embedding = embeddings[neighbors]
            negative_samples_embedding = embeddings[negative_samples]
            loss_list = torch.zeros((2 * negative_size), dtype=torch.float32)
            ind = 0
            for h2 in neighbors_embedding:
                neighbor_loss = 1 - StructurePredictor.cosine_similarity(h1, h2)
                loss_list[ind] += neighbor_loss
                ind += 1

            for h2 in negative_samples_embedding:
                negative_loss = StructurePredictor.cosine_similarity(h1, h2)
                loss_list[ind] += negative_loss
                ind += 1

            if loss_list.shape[0] > 0:
                node_loss = loss_list.mean()
            else:
                node_loss = torch.tensor([0], dtype=torch.float32)
            loss[node_index] = node_loss

        return loss

    def train(self, subgraphs=[]):
        self.model.train()
        out = self.model(subgraphs, self.graph)
        loss = self.calc_loss(out[f"structure_model"])

        return out, loss

    def fit(
        self,
        epochs=config.gen_epochs,
        plot=False,
        bar=False,
    ):
        if bar:
            bar = tqdm(total=epochs, position=0)
        res = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            _, loss = self.train()

            train_loss = loss[self.graph.train_mask].mean()
            val_loss = loss[self.graph.test_mask].mean()

            train_loss.backward()
            self.optimizer.step()

            metrics = {
                "Train Loss": round(train_loss.item(), 4),
                "Val Loss": round(val_loss.item(), 4),
            }

            if bar:
                bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
                bar.set_postfix(metrics)
                bar.update(1)

            metrics["Epoch"] = epoch + 1
            res.append(metrics)

        # Test
        # test_accuracy = test(self.GNN, self.prepared_data)
        if plot:
            dataset = pd.DataFrame.from_dict(res)
            dataset.set_index("Epoch", inplace=True)
            dataset[
                [
                    "Train Loss",
                    "Val Loss",
                ]
            ].plot()
            plt.title(f"classifier loss client {self.id}")

        return res

    def joint_train(self, clients, epochs=1):
        metrics = {}
        subgraphs = []
        for client in clients:
            subgraphs.append(client.subgraph)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss_list = torch.zeros(len(subgraphs), dtype=torch.float32)
            out, structure_loss = self.train(subgraphs)
            for ind, client in enumerate(clients):
                node_ids = client.get_nodes()
                y = client.subgraph.y
                train_mask = client.subgraph.train_mask
                val_mask = client.subgraph.val_mask

                client_structure_loss = structure_loss[node_ids]
                y_pred = out[f"client{client.id}"]

                cls_train_loss = self.criterion(y_pred[train_mask], y[train_mask])
                str_train_loss = client_structure_loss[train_mask].mean()
                train_loss = cls_train_loss + str_train_loss
                loss_list[ind] = train_loss

                train_acc = calc_accuracy(
                    y_pred[train_mask].argmax(dim=1),
                    y[train_mask],
                )

                # Validation
                self.model.eval()
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
                }

                metrics[f"client{client.id}"] = result

            total_loss = loss_list.mean()
            total_loss.backward()
            self.optimizer.step()
            metrics["Total Loss"] = round(total_loss.item(), 4)
            metrics["Structure Loss"] = round(structure_loss.mean().item(), 4)

        return metrics
