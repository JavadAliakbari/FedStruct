import logging
import operator
from ast import List
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset, Planetoid

from src.utils.config_parser import Config
from src.utils.graph import Graph
from src.utils.graph_partinioning import louvain_graph_cut
from src.models.GNN_models import GNN

config = Config()


class JointModel(torch.nn.Module):
    """Joint Model"""

    def __init__(
        self,
        clients,
        structure_layer_sizes,
        client_layer_sizes=None,
        dropout=config.model.dropout,
        linear_layer=False,
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
            assert len(client_layer_sizes) - 1 == len(
                structure_layer_sizes
            ), "Number of layers must be equal"
            self.num_clients = num_clients
            out_dims = client_layer_sizes
            in_dims = list(map(operator.add, client_layer_sizes, structure_layer_sizes))

            for id in range(self.num_clients):
                model = GNN(
                    in_dims=in_dims,
                    out_dims=out_dims,
                    dropout=dropout,
                    linear_layer=linear_layer,
                    last_layer=last_layer,
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

    def __getitem__(self, item):
        return self.models[item]

    def __setitem__(self, key, item):
        self.models[key] = item

    def __repr__(self):
        return repr(self.models)

    def __len__(self):
        return len(self.models)

    def __iter__(self):
        return iter(self.models)

    def keys(self):
        return self.models.keys()

    def values(self):
        return self.models.values()

    def items(self):
        return self.models.items()

    def pop(self, *args):
        return self.models.pop(*args)

    def reset_parameters(self) -> None:
        for model in self.models.values():
            model.reset_parameters()

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
            H.append(subgraphs[client_id].x)

        S = structure_graph.structural_features

        for layer_id in range(self.num_layers):
            for client_id in range(num_subgraphs):
                node_ids = subgraphs[client_id].node_ids
                edge_index = subgraphs[client_id].edge_index
                x_s = S[node_ids]
                x = torch.hstack((H.popleft(), x_s))
                h = self.models[f"client{client_id}"][layer_id](x, edge_index).relu()
                h = F.dropout(h, p=self.dropout, training=self.training)
                H.append(h)

            edge_index = structure_graph.edge_index
            model = self.models["structure_model"][layer_id]
            S = model(S, edge_index)

            if layer_id != self.num_layers - 1:
                S = nn.functional.relu(S)
                S = F.dropout(S, p=self.dropout, training=self.training)

        for client_id in range(num_subgraphs):
            node_ids = subgraphs[client_id].node_ids
            edge_index = subgraphs[client_id].edge_index
            x_s = S[node_ids]
            x = torch.hstack((H.popleft(), x_s))
            model = self.models[f"client{client_id}"]
            if model.linear_layer:
                h = model[self.num_layers](x)
            else:
                h = model[self.num_layers](x, edge_index)
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
    graph.add_structural_features(config.structure_model.num_structural_features)

    num_classes = dataset.num_classes

    subgraphs = louvain_graph_cut(graph)
    num_clients = len(subgraphs)

    GNN = JointModel(
        num_clients=num_clients,
        client_layer_sizes=[graph.num_features] + config.model.gnn_layer_sizes,
        structure_layer_sizes=[config.structure_model.num_structural_features]
        + config.model.gnn_layer_sizes,
        num_classes=num_classes,
    )

    criterion = torch.nn.CrossEntropyLoss()
    parameters = list(GNN.parameters())
    optimizer = torch.optim.Adam(
        parameters, lr=config.model.lr, weight_decay=config.model.weight_decay
    )

    GNN.forward(subgraphs, graph)
    a = 2
