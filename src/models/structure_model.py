from ast import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.datasets import TUDataset, Planetoid

from src.utils import config
from src.utils.graph import Graph
from src.utils.graph_partinioning import louvain_graph_cut


class JointModel(torch.nn.Module):
    """Joint Model"""

    def __init__(
        self,
        num_clients,
        client_layer_sizes,
        structure_layer_sizes,
        num_classes,
        dropout=0.5,
        last_layer="softmax",
    ):
        super().__init__()
        assert len(client_layer_sizes) == len(
            structure_layer_sizes
        ), "Number of layers must be equal"

        self.num_clients = num_clients
        self.num_layers = len(structure_layer_sizes) - 1

        self.dropout = dropout
        self.last_layer = last_layer
        self.models = nn.ParameterDict()

        for id in range(num_clients):
            layers = self.create_models(client_layer_sizes, structure_layer_sizes)

            output_layer = nn.Linear(
                client_layer_sizes[-1] + structure_layer_sizes[-1], num_classes
            )
            layers.append(output_layer)
            self.models[f"client{id}"] = layers

        self.models[f"structure_model"] = self.create_models(structure_layer_sizes)

    def create_models(self, layer_size1, layer_size2=None):
        if layer_size2 is None:
            layer_size2 = self.num_layers * [0]
        layers = nn.ParameterList()
        for layer_num in range(self.num_layers):
            layer = SAGEConv(
                layer_size1[layer_num] + layer_size2[layer_num],
                layer_size1[layer_num + 1],
            )
            layers.append(layer)

        return layers

    def reset_parameters(self) -> None:
        for layers in self.models.values():
            for layer in layers:
                layer.reset_parameters()

    def state_dict(self):
        model_weights = {}
        for model_name, layers in self.models.items():
            layer_weights = {}
            for id, layer in enumerate(layers):
                layer_weights[f"layer{id}"] = layer.state_dict()

            model_weights[model_name] = layer_weights
        return model_weights

    def load_state_dict(self, model_weights: dict) -> None:
        for model_name, layers in self.models.items():
            for id, layer in enumerate(layers):
                layer.load_state_dict(model_weights[model_name][f"layer{id}"])

    def forward(self, subgraphs: List, structure_graph: Graph):
        assert len(subgraphs) <= self.num_clients, "There is not any models as data"

        num_subgraphs = len(subgraphs)

        H = []
        for client_id in range(num_subgraphs):
            H.append(subgraphs[client_id].x)

        S = structure_graph.structural_features

        for layer_id in range(self.num_layers):
            edge_index = structure_graph.edge_index
            s = self.models["structure_model"][layer_id](S, edge_index)
            if layer_id != self.num_layers - 1:
                s = nn.functional.relu(s)
                s = F.dropout(s, p=self.dropout, training=self.training)

            for client_id in range(num_subgraphs):
                node_ids = subgraphs[client_id].node_ids
                edge_index = subgraphs[client_id].edge_index
                x = torch.hstack((H[client_id], S[node_ids]))
                h = self.models[f"client{client_id}"][layer_id](x, edge_index).relu()
                h = F.dropout(h, p=self.dropout, training=self.training)
                H[client_id] = h

            S = s

        for client_id in range(num_subgraphs):
            node_ids = subgraphs[client_id].node_ids
            x = torch.hstack((H[client_id], S[node_ids]))
            edge_index = subgraphs[client_id].edge_index
            h = self.models[f"client{client_id}"][self.num_layers](x)
            H[client_id] = h

        out = {}
        out[f"structure_model"] = S
        for client_id in range(num_subgraphs):
            x = H[client_id]
            if self.last_layer == "softmax":
                out[f"client{client_id}"] = F.softmax(x, dim=1)
            elif self.last_layer == "relu":
                out[f"client{client_id}"] = F.relu(x)
            else:
                out[f"client{client_id}"] = x

        return out


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
