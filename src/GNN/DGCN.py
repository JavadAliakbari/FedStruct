import os

import torch
from torch_geometric.loader import NeighborLoader

from src.GNN.sGNN import SGNNMaster
from src.utils.graph import AGraph
from src.classifier import Classifier
from src.utils.config_parser import Config
from src.models.model_binders import (
    ModelBinder,
    ModelSpecs,
)

dev = os.environ.get("device", "cpu")
device = torch.device(dev)

path = os.environ.get("CONFIG_PATH")
config = Config(path)


class DGCN(Classifier):
    def __init__(
        self, graph: AGraph, hidden_layer_size=config.feature_model.DGCN_layer_sizes
    ):
        super().__init__()
        self.graph: AGraph = graph
        self.create_model(hidden_layer_size)
        # self.create_model(config.feature_model.DGCN_layer_sizes)

    def create_model(self, hidden_layer_size=[]):
        layer_sizes = (
            [self.graph.num_features] + hidden_layer_size + [self.graph.num_classes]
        )

        model_specs = [
            ModelSpecs(
                type="MLP",
                layer_sizes=layer_sizes,
                final_activation_function="linear",
                normalization="layer",
            ),
        ]

        self.model: ModelBinder = ModelBinder(model_specs)
        self.model.to(device)

    def get_embeddings(self):
        H = self.model(self.graph.x)
        if self.graph.abar.is_sparse and H.device.type == "mps":
            H = self.graph.abar.matmul(H.cpu()).to(device)
        else:
            H = torch.matmul(self.graph.abar, H)
        return H


class SDGCN(DGCN):
    def __init__(
        self,
        graph: AGraph,
        hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
    ):
        super().__init__(graph, hidden_layer_size)

    def parameters(self):
        parameters = super().parameters()
        if self.graph.x.requires_grad:
            parameters += [self.graph.x]

        return parameters

    def get_grads(self, just_SFV=False):
        grads = super().get_grads(just_SFV)
        if self.graph.x is not None:
            if self.graph.x.requires_grad:
                grads["SFV"] = [self.graph.x.grad]

        return grads

    def set_grads(self, grads):
        super().set_grads(grads)
        if "SFV" in grads.keys():
            self.graph.x.grad = grads["SFV"][0]

    def zero_grad(self, set_to_none=False):
        super().zero_grad(set_to_none)
        if self.graph.x.requires_grad:
            self.graph.x.grad = None


class SDGCNMaster(SGNNMaster):
    def __init__(
        self,
        graph: AGraph,
        hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
    ):
        super().__init__(graph)
        self.GNN_structure_embedding = None
        self.create_model(hidden_layer_size)

    def create_model(self, hidden_layer_size=[]):
        layer_sizes = (
            [self.graph.num_features] + hidden_layer_size + [self.graph.num_classes]
        )

        model_specs = [
            ModelSpecs(
                type="MLP",
                layer_sizes=layer_sizes,
                final_activation_function="linear",
                normalization="layer",
            ),
        ]

        self.model: ModelBinder = ModelBinder(model_specs)
        self.model.to(device)

    def get_embeddings(self, node_ids):
        if self.GNN_structure_embedding is None:
            H = self.model(self.graph.x)
            if self.graph.abar.is_sparse and H.device.type == "mps":
                self.GNN_structure_embedding = self.graph.abar.matmul(H.cpu()).to(
                    device
                )
            else:
                self.GNN_structure_embedding = torch.matmul(self.graph.abar, H)
        return self.GNN_structure_embedding[node_ids]
