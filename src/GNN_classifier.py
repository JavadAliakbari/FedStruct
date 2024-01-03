import os
from ast import List

import torch
from torch_geometric.loader import NeighborLoader

from src.utils.utils import *
from src.utils.graph import Graph
from src.classifier import Classifier
from src.utils.config_parser import Config
from src.models.GNN_models import (
    ModelBinder,
    ModelSpecs,
)

path = os.environ.get("CONFIG_PATH")
config = Config(path)


class GNNClassifier(Classifier):
    def __init__(
        self,
        id,
        num_classes,
        save_path="./",
        logger=None,
    ):
        super().__init__(
            id=id,
            num_classes=num_classes,
            save_path=save_path,
            logger=logger,
        )

        self.abar = None
        self.abar_i = None

        self.GNN_structure_embedding = None
        self.get_structure_embeddings_from_server = None

    def reset(self):
        super().reset()

        self.GNN_structure_embedding = None

    def restart(self):
        super().restart()
        self.abar = None
        self.abar_i = None
        self.GNN_structure_embedding = None
        self.get_structure_embeddings_from_server = None

    def set_GNN_FPM(self, dim_in=None):
        if dim_in is None:
            dim_in = self.graph.num_features

        gnn_layer_sizes = [dim_in] + config.feature_model.gnn_layer_sizes
        mlp_layer_sizes = (
            [config.feature_model.gnn_layer_sizes[-1]]
            # + config.feature_model.mlp_layer_sizes
            + [self.num_classes]
        )

        model_specs = [
            ModelSpecs(
                type="GNN",
                layer_sizes=gnn_layer_sizes,
                final_activation_function="linear",
                # final_activation_function="relu",
                normalization="layer",
                # normalization="batch",
            ),
            ModelSpecs(
                type="MLP",
                layer_sizes=mlp_layer_sizes,
                final_activation_function="linear",
                normalization=None,
            ),
        ]

        self.feature_model: ModelBinder = ModelBinder(model_specs)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.feature_model.parameters(),
            lr=config.model.lr,
            weight_decay=config.model.weight_decay,
        )

        self.abar_i = None

    def set_DGCN_FPM(self, dim_in=None):
        if dim_in is None:
            dim_in = self.graph.num_features

        mlp_layer_sizes = (
            [dim_in] + config.feature_model.desicion_layer_sizes + [self.num_classes]
        )

        model_specs = [
            ModelSpecs(
                type="MLP",
                layer_sizes=mlp_layer_sizes,
                final_activation_function="linear",
                normalization="layer",
            ),
        ]

        self.feature_model: ModelBinder = ModelBinder(model_specs)

        self.abar_i = obtain_a(
            self.graph.edge_index,
            self.graph.num_nodes,
            config.feature_model.mp_layers,
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.feature_model.parameters(),
            lr=config.model.lr,
            weight_decay=config.model.weight_decay,
        )

    def set_abar(self, abar):
        self.abar = abar

    def set_GNN_SPM(self, dim_in=None, get_structure_embeddings=None):
        if self.id == "Server":
            if dim_in is None:
                dim_in = self.graph.num_structural_features

            gnn_layer_sizes = [
                dim_in
            ] + config.structure_model.GNN_structure_layers_sizes
            mlp_layer_sizes = (
                [config.structure_model.GNN_structure_layers_sizes[-1]]
                # + config.feature_model.mlp_layer_sizes
                + [self.num_classes]
            )

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

            self.structure_model: ModelBinder = ModelBinder(model_specs)
            self.optimizer.add_param_group(
                {"params": self.structure_model.parameters()}
            )
        else:
            self.structure_model = None
            self.get_structure_embeddings_from_server = get_structure_embeddings

        self.abar = None

    def set_DGCN_SPM(self, dim_in=None):
        if dim_in is None:
            dim_in = config.structure_model.num_structural_features
        SPM_layer_sizes = (
            [dim_in]
            + config.structure_model.MP_structure_layers_sizes
            + [self.num_classes]
        )
        model_specs = [
            ModelSpecs(
                type="MLP",
                layer_sizes=SPM_layer_sizes,
                final_activation_function="linear",
                normalization="layer",
            )
        ]

        self.structure_model: ModelBinder = ModelBinder(model_specs)
        self.optimizer.add_param_group({"params": self.structure_model.parameters()})

    def get_structure_embeddings2(self, node_ids):
        if self.GNN_structure_embedding is None:
            x = self.SFV
            edge_index = self.graph.edge_index
            self.GNN_structure_embedding = self.structure_model(x, edge_index)

        return self.GNN_structure_embedding[node_ids]

    def prepare_data(
        self,
        graph: Graph,
        batch_size: int = 16,
        num_neighbors: List = [5, 10],
    ):
        self.graph = graph
        if self.graph.train_mask is None:
            self.graph.add_masks()

        # self.data_loader = NeighborLoader(
        #     self.graph,
        #     num_neighbors=num_neighbors,
        #     batch_size=batch_size,
        #     # input_nodes=self.graph.train_mask,
        #     shuffle=True,
        # )

    def set_SFV(self, SFV):
        self.SFV = deepcopy(SFV)
        if self.SFV.requires_grad:
            self.optimizer.add_param_group({"params": self.SFV})

    def get_embeddings(model, x, edge_index=None, a=None):
        H = model(x, edge_index)
        if a is not None:
            H = a.matmul(H)
        return H

    @torch.no_grad()
    def calc_test_accuracy(self, metric="acc"):
        self.feature_model.eval()
        if self.structure_model is not None:
            self.structure_model.eval()

        y_pred = self.get_prediction()
        y = self.graph.y
        test_mask = self.graph.test_mask

        test_loss, test_acc, test_f1_score = calc_metrics(y, y_pred, test_mask)

        if metric == "acc":
            return test_acc
        elif metric == "f1":
            return test_f1_score
        else:
            return test_loss

    def get_prediction(self):
        h = GNNClassifier.get_embeddings(
            self.feature_model,
            self.graph.x,
            self.graph.edge_index,
            self.abar_i,
        )

        if self.structure_model is not None:
            s = GNNClassifier.get_embeddings(
                self.structure_model,
                self.SFV,
                self.graph.edge_index,
                a=self.abar,
            )
            o = h + s
        elif self.get_structure_embeddings_from_server is not None:
            s = self.get_structure_embeddings_from_server(self.graph.node_ids)
            o = h + s
        elif self.SFV is not None:
            if self.SFV.requires_grad:
                s = self.abar.matmul(self.SFV)
                o = h + s
        else:
            o = h
        y_pred = torch.nn.functional.softmax(o, dim=1)

        return y_pred

    def train_step(self, scale=False):
        y_pred = self.get_prediction()
        y = self.graph.y

        train_mask, val_mask, _ = self.graph.get_masks()

        train_loss, train_acc, train_f1_score = calc_metrics(y, y_pred, train_mask)
        val_loss, val_acc, val_f1_score = calc_metrics(y, y_pred, val_mask)

        if scale:
            train_loss *= self.graph.num_nodes
        train_loss.backward(retain_graph=True)

        return train_loss, train_acc, train_f1_score, val_loss, val_acc, val_f1_score
