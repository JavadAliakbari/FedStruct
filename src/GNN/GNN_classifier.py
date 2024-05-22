import os
from ast import List

import torch
from torch_geometric.loader import NeighborLoader

from src.utils.utils import *
from src.utils.graph import Graph
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


class GNNClassifier(Classifier):
    def __init__(
        self,
        id,
        save_path="./",
        logger=None,
    ):
        super().__init__(
            id=id,
            save_path=save_path,
            logger=logger,
        )

        self.abar = None
        self.abar_i = None

        self.GNN_structure_embedding = None
        self.get_structure_embeddings_from_server = None
        self.data_type = None

    def reset(self):
        super().reset()

        self.GNN_structure_embedding = None

    def restart(self):
        super().restart()
        self.abar = None
        self.abar_i = None
        self.GNN_structure_embedding = None
        self.get_structure_embeddings_from_server = None
        self.data_type = None

    def set_GNN_FPM(self):
        gnn_layer_sizes = [
            self.graph.num_features
        ] + config.feature_model.gnn_layer_sizes
        mlp_layer_sizes = [config.feature_model.gnn_layer_sizes[-1]] + [
            self.graph.num_classes
        ]

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
        self.feature_model.to(device)

        self.optimizer = torch.optim.Adam(
            self.feature_model.parameters(),
            lr=config.model.lr,
            weight_decay=config.model.weight_decay,
        )

        self.abar_i = None

    def set_DGCN_FPM(self, dim_in=None, use_abar=True):
        mlp_layer_sizes = (
            [self.graph.num_features]
            + config.feature_model.DGCN_layer_sizes
            + [self.graph.num_classes]
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
        self.feature_model.to(device)

        self.abar_i = None
        if (
            use_abar
        ):  # both of them have equivalent performance. maybe speed is a little bit different
            self.abar_i = obtain_a(
                self.graph.edge_index,
                self.graph.num_nodes,
                config.feature_model.DGCN_layers,
            )
        else:
            model_specs.append(
                ModelSpecs(
                    type="DGCN",
                    num_layers=config.feature_model.DGCN_layers,
                    final_activation_function="linear",
                )
            )

        self.optimizer = torch.optim.Adam(
            self.feature_model.parameters(),
            lr=config.model.lr,
            weight_decay=config.model.weight_decay,
        )

    def set_abar(self, abar):
        self.abar = abar

    def set_structure_embedding(self, get_structure_embeddings):
        self.get_structure_embeddings_from_server = get_structure_embeddings

    def set_GNN_SPM(self, dim_in=None):
        # if self.id == "Server":
        if dim_in is None:
            dim_in = self.graph.num_structural_features

        gnn_layer_sizes = [dim_in] + config.structure_model.GNN_structure_layers_sizes
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

        self.structure_model: ModelBinder = ModelBinder(model_specs)
        self.structure_model.to(device)
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.structure_model.parameters(),
                lr=config.model.lr,
                weight_decay=config.model.weight_decay,
            )
        else:
            self.optimizer.add_param_group(
                {"params": self.structure_model.parameters()}
            )

        self.abar = None

    def set_DGCN_SPM(self, dim_in=None):
        if dim_in is None:
            dim_in = config.structure_model.num_structural_features
        SPM_layer_sizes = (
            [dim_in]
            + config.structure_model.DGCN_structure_layers_sizes
            + [self.graph.num_classes]
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
        self.structure_model.to(device)
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.structure_model.parameters(),
                lr=config.model.lr,
                weight_decay=config.model.weight_decay,
            )
        else:
            self.optimizer.add_param_group(
                {"params": self.structure_model.parameters()}
            )

    def get_structure_embeddings2(self, node_ids):
        if self.GNN_structure_embedding is None:
            x = self.SFV
            edge_index = self.graph.edge_index
            self.GNN_structure_embedding = self.structure_model(x, edge_index)

        return self.GNN_structure_embedding[node_ids]

    def prepare_data(
        self,
        graph: Graph,
        data_type="feature",
        batch_size: int = 16,
        num_neighbors: List = [5, 10],
    ):
        self.data_type = data_type
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
        # self.SFV = deepcopy(SFV)
        self.SFV = torch.tensor(
            SFV.detach().cpu().numpy(),
            requires_grad=SFV.requires_grad,
            device=device,
        )
        if self.SFV.requires_grad:
            if self.optimizer is None:
                self.optimizer = torch.optim.Adam(
                    [self.SFV],
                    lr=config.model.lr,
                    weight_decay=config.model.weight_decay,
                )
            else:
                self.optimizer.add_param_group({"params": self.SFV})

    def get_embeddings(model, x, edge_index=None, a=None):
        H = model(x, edge_index)
        if a is not None:
            if not a.is_sparse:
                H = torch.matmul(a, H)
            else:
                if dev != "mps":
                    H = torch.matmul(a, H)
                else:
                    H = a.matmul(H.to("cpu")).to(device)
        return H

    @torch.no_grad()
    def calc_test_accuracy(self, metric="acc"):
        self.eval()

        y = self.graph.y
        test_mask = self.graph.test_mask
        y_pred = self.get_prediction(model_use=self.data_type)

        test_loss, test_acc = calc_metrics(y, y_pred, test_mask)
        if self.data_type == "f+s":
            y_pred_f = self.get_prediction(model_use="feature")
            y_pred_s = self.get_prediction(model_use="structure")
            test_loss_f, test_acc_f = calc_metrics(y, y_pred_f, test_mask)
            test_loss_s, test_acc_s = calc_metrics(y, y_pred_s, test_mask)
        else:
            test_acc_f, test_loss_f, test_acc_s, test_loss_s = None, None, None, None

        if metric == "acc":
            return test_acc, test_acc_f, test_acc_s
        # elif metric == "f1":
        #     return test_f1_score
        else:
            return test_loss, test_loss_f, test_loss_s

    def get_feature_embeddings(self):
        h = GNNClassifier.get_embeddings(
            self.feature_model,
            self.graph.x,
            self.graph.edge_index,
            self.abar_i,
        )

        return h

    def get_structure_embeddings(self):
        s = None
        if self.structure_model is not None:
            s = GNNClassifier.get_embeddings(
                model=self.structure_model,
                x=self.SFV,
                a=self.abar,
            )
        elif self.get_structure_embeddings_from_server is not None:
            s = self.get_structure_embeddings_from_server(self.graph.node_ids)
        elif self.SFV is not None:
            if self.SFV.requires_grad:
                s = self.abar.matmul(self.SFV)

        return s

    def get_prediction(self, model_use="f+s"):
        y_pred = None
        o = None
        h = None
        s = None
        if model_use in ["feature", "f+s"]:
            h = self.get_feature_embeddings()

        if model_use in ["structure", "f+s"]:
            s = self.get_structure_embeddings()

        if h is not None:
            o = h

        if s is not None:
            if o is not None:
                o += s
            else:
                o = s

        if o is not None:
            y_pred = torch.nn.functional.softmax(o, dim=1)

        return y_pred

    def train_step(self, eval_=True):
        y_pred = self.get_prediction(model_use=self.data_type)
        y = self.graph.y

        train_mask, val_mask, _ = self.graph.get_masks()

        train_loss, train_acc = calc_metrics(y, y_pred, train_mask)
        train_loss.backward(retain_graph=True)

        if eval_:
            self.eval()
            y_pred_val = self.get_prediction(model_use=self.data_type)
            val_loss, val_acc = calc_metrics(y, y_pred_val, val_mask)

            if self.data_type == "f+s":
                y_pred_f = self.get_prediction(model_use="feature")
                y_pred_s = self.get_prediction(model_use="structure")
                _, val_acc_f = calc_metrics(y, y_pred_f, val_mask)
                _, val_acc_s = calc_metrics(y, y_pred_s, val_mask)
            else:
                val_acc_f, val_acc_s = 0, 0

            return train_loss, train_acc, val_loss, val_acc, val_acc_f, val_acc_s
        else:
            return train_loss, train_acc, torch.tensor(0), 0, 0, 0
