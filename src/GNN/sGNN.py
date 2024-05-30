import torch
from torch_geometric.loader import NeighborLoader

from src import *
from src.utils.graph import Graph
from src.classifier import Classifier
from src.models.model_binders import (
    ModelBinder,
    ModelSpecs,
)


class SGNNSlave(Classifier):
    def __init__(self, graph: Graph, server_embedding_func):
        super().__init__()
        self.graph = graph
        self.__set_embedding_func(server_embedding_func)

    def __set_embedding_func(self, server_embedding_func):
        self.get_embeddings_ = server_embedding_func

    def get_embeddings(self, node_ids):
        return self.get_embeddings_(node_ids)

    def __call__(self):
        return self.get_embeddings(self.graph.node_ids)

    def get_prediction(self):
        s = self.get_embeddings(self.graph.node_ids)
        y_pred = torch.nn.functional.softmax(s, dim=1)

        return y_pred


class SGNNMaster(SGNNSlave):
    def __init__(self, graph: Graph):
        super().__init__(graph, None)
        self.GNN_structure_embedding = None
        self.create_model()

    def reset(self):
        super().reset()
        self.GNN_structure_embedding = None

    def restart(self):
        super().restart()
        self.GNN_structure_embedding = None

    def parameters(self):
        parameters = super().parameters()
        if self.graph.x is not None:
            if self.graph.x.requires_grad:
                parameters += [self.graph.x]

        return parameters

    def get_grads(self, just_SFV=False):
        grads = super().get_grads(just_SFV)
        if self.graph.x.requires_grad:
            grads["SFV"] = [self.graph.x.grad]

        return grads

    def zero_grad(self, set_to_none=False):
        super().zero_grad(set_to_none)
        if self.graph.x.requires_grad:
            self.graph.x.grad = None

    def create_model(self):
        gnn_layer_sizes = [
            self.graph.num_features
        ] + config.structure_model.GNN_structure_layers_sizes
        mlp_layer_sizes = [config.structure_model.GNN_structure_layers_sizes[-1]] + [
            self.graph.num_classes
        ]

        model_specs = [
            ModelSpecs(
                type="GNN",
                layer_sizes=gnn_layer_sizes,
                final_activation_function="linear",
                # final_activation_function="relu",
                # normalization="layer",
                normalization="batch",
            ),
            ModelSpecs(
                type="MLP",
                layer_sizes=mlp_layer_sizes,
                final_activation_function="linear",
                normalization=None,
            ),
        ]

        self.model: ModelBinder = ModelBinder(model_specs)
        self.model.to(device)

    def get_embeddings(self, node_ids):
        if self.GNN_structure_embedding is None:
            x = self.graph.x
            edge_index = self.graph.edge_index
            self.GNN_structure_embedding = self.model(x, edge_index)

        return self.GNN_structure_embedding[node_ids]

    def __call__(self, node_ids=None):
        if node_ids is None:
            return self.get_embeddings(self.graph.node_ids)
        else:
            return self.get_embeddings(node_ids)
