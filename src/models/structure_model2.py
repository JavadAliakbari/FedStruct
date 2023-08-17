import logging
from ast import List
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset, Planetoid

from src.utils.graph import Graph
from src.utils.config_parser import Config
from src.utils.graph_partinioning import louvain_graph_cut
from src.models.GNN_models import GNN, MLP, calc_accuracy

config = Config()


class JointModel(torch.nn.Module):
    """Joint Model"""

    def __init__(
        self,
        clients,
        structure_layer_sizes,
        num_classes=None,
        client_layer_sizes=None,
        dropout=config.model.dropout,
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
                self.local_sd_model = GNN(
                    in_dims=client_layer_sizes,
                    decision_layer=True,
                    dropout=config.model.dropout,
                    last_layer="linear",
                )
                # model = MLP(
                #     layer_sizes=client_layer_sizes,
                #     dropout=dropout,
                #     softmax=False,
                # )

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
            self.models[f"linear_layer"] = nn.Linear(
                self.models[f"client{id}"][-1].out_features + structure_layer_sizes[-1],
                num_classes,
            )

        # self.linear_layers = nn.ParameterDict()
        # for id in range(self.num_clients):
        #     linear = nn.Linear(
        #         self.models[f"client{id}"][-1].out_features + structure_layer_sizes[-1],
        #         num_classes,
        #     )
        #     self.linear_layers[f"client{id}"] = linear

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
            x = subgraphs[client_id].x
            edge_index = subgraphs[client_id].edge_index
            h = self.models[f"client{client_id}"](x, edge_index)
            H.append(h)

        x = structure_graph.structural_features
        edge_index = structure_graph.edge_index
        model = self.models["structure_model"]
        S = model(x, edge_index)

        for client_id in range(num_subgraphs):
            node_ids = subgraphs[client_id].node_ids
            x_s = S[node_ids]
            x = torch.hstack((H.popleft(), x_s))
            h = self.models[f"linear_layer"](x)
            # h = self.linear_layers[f"client{client_id}"](x)
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
    graph.add_structural_features(
        structure_type=config.structure_model.structure_type,
        num_structural_features=config.structure_model.num_structural_features,
    )

    num_classes = dataset.num_classes

    subgraphs = louvain_graph_cut(graph)
    num_clients = len(subgraphs)

    model = JointModel(
        clients=num_clients,
        client_layer_sizes=[graph.num_features] + config.model.gnn_layer_sizes,
        structure_layer_sizes=[config.structure_model.num_structural_features]
        + config.structure_model.structure_layers_sizes,
        num_classes=num_classes,
    )

    criterion = torch.nn.CrossEntropyLoss()
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(
        parameters, lr=config.model.lr, weight_decay=config.model.weight_decay
    )

    out = model.forward(subgraphs, graph)
    a = 2
