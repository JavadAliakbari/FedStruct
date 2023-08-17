import logging
from collections import deque

import torch
from torch_geometric.datasets import TUDataset, Planetoid
from src.client import Client

from src.utils.graph import Graph
from src.utils.config_parser import Config
from src.utils.graph_partinioning import louvain_graph_cut

config = Config()


class JointModel(torch.nn.Module):
    """Joint Model"""

    def __init__(
        self,
        server,
        clients,
        logger=None,
    ):
        super().__init__()
        self.clients = clients
        self.server = server
        self.LOGGER = logger or logging

    def __getitem__(self, item):
        return self.clients[item]

    def __repr__(self):
        return repr(self.clients)

    def __len__(self):
        return len(self.clients)

    def __iter__(self):
        return iter(self.clients)

    def reset_parameters(self) -> None:
        self.server.reset_parameters()
        for client in self.clients:
            client.reset_parameters()

    def state_dict(self):
        model_weights = {}
        for client in self.clients:
            weights = client.state_dict()
            model_weights[f"client{client.id}"] = weights

        return model_weights

    def load_state_dict(self, model_weights: dict) -> None:
        for client in self.clients:
            weights = model_weights[f"client{client.id}"]
            client.load_state_dict(weights)

    def forward(self):
        out = {}
        S = self.server.get_structure_embeddings()
        out[f"structure_model"] = S

        H = deque()
        for client in self.clients:
            h = client.get_feature_embeddings()
            H.append(h)

        for client in self.clients:
            node_ids = client.get_nodes()
            x_s = S[node_ids]
            x = torch.hstack((H.popleft(), x_s))
            out[f"client{client.id}"] = client.predict_labels(x)

        return out

    def step(self, client: Client, train=True):
        S = self.server.get_structure_embeddings(train=train)
        h = client.get_feature_embeddings()
        node_ids = client.get_nodes()
        x_s = S[node_ids]
        x = torch.hstack((h, x_s))
        out = client.predict_labels(x)

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
