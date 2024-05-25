import os

import torch
from torch_geometric.loader import NeighborLoader

from src.utils.utils import *
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


class FDGCN(Classifier):
    def __init__(self, graph: AGraph):
        super().__init__()
        self.graph = graph
        self.create_model()

    def create_model(self):
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

        self.model: ModelBinder = ModelBinder(model_specs)
        self.model.to(device)

    def get_embeddings(self):
        H = self.model(self.graph.x)
        if self.graph.abar.is_sparse and H.device.type == "mps":
            H = self.graph.abar.matmul(H.cpu()).to(device)
        else:
            H = torch.matmul(self.graph.abar, H)
        return H

    def __call__(self):
        return self.get_embeddings()

    def get_prediction(self):
        H = self.get_embeddings()
        y_pred = torch.nn.functional.softmax(H, dim=1)
        return y_pred

    def train_step(self, eval_=True):
        y_pred = self.get_prediction()
        y = self.graph.y

        train_mask, val_mask, _ = self.graph.get_masks()

        train_loss, train_acc = calc_metrics(y, y_pred, train_mask)
        train_loss.backward()

        if eval_:
            self.eval()
            y_pred_val = self.get_prediction()
            val_loss, val_acc = calc_metrics(y, y_pred_val, val_mask)

            return train_loss.item(), train_acc, val_loss.item(), val_acc
        else:
            return train_loss.item(), train_acc, 0, 0

    @torch.no_grad()
    def calc_test_accuracy(self, metric="acc"):
        self.eval()

        y = self.graph.y
        test_mask = self.graph.test_mask
        y_pred = self.get_prediction()

        test_loss, test_acc = calc_metrics(y, y_pred, test_mask)

        if metric == "acc":
            return (test_acc,)
        # elif metric == "f1":
        #     return test_f1_score
        else:
            return (test_loss.item(),)
