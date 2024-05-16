import os

import torch
from torch.utils.data import DataLoader

from src.utils.graph import Data
from src.classifier import Classifier
from src.utils.config_parser import Config
from src.utils.utils import calc_metrics, calc_accuracy, calc_f1_score
from src.MLP.MLP_model import MLP

dev = os.environ.get("device", "cpu")
device = torch.device(dev)
cpu_device = torch.device("cpu")

path = os.environ.get("CONFIG_PATH")
config = Config(path)


class MLPClassifier(Classifier):
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

    def set_classifiers(self, dim_in=None):
        if dim_in is None:
            dim_in = self.data.num_features

        layer_sizes = (
            [dim_in] + config.feature_model.mlp_layer_sizes + [self.data.num_classes]
        )
        self.feature_model = MLP(
            layer_sizes=layer_sizes,
            last_layer="linear",
            dropout=config.model.dropout,
            normalization="layer",
            # normalization="batch",
        )
        self.feature_model.to(device)

        self.optimizer = torch.optim.Adam(
            self.feature_model.parameters(),
            lr=config.model.lr,
            weight_decay=config.model.weight_decay,
        )

    def prepare_data(
        self,
        data: Data,
        batch_size: int = 16,
    ):
        self.data = data
        if self.data.train_mask is None:
            self.data.add_masks()

        train_x = self.data.x[self.data.train_mask]
        train_y = self.data.y[self.data.train_mask]
        val_x = self.data.x[self.data.val_mask]
        val_y = self.data.y[self.data.val_mask]
        test_x = self.data.x[self.data.test_mask]
        test_y = self.data.y[self.data.test_mask]

        self.train_loader = DataLoader(
            list(zip(train_x, train_y)), batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            list(zip(val_x, val_y)), batch_size=batch_size, shuffle=False
        )

        self.test_data = [test_x, test_y]

    @torch.no_grad()
    def calc_test_accuracy(self, metric="acc"):
        self.feature_model.eval()
        test_x, test_y = self.test_data
        out = self.feature_model(test_x)
        if metric == "acc":
            val_acc = calc_accuracy(out.argmax(dim=1), test_y)
        elif metric == "f1":
            val_acc = calc_f1_score(out.argmax(dim=1), test_y)

        return val_acc, None, None

    def get_prediction(self):
        h = self.feature_model(self.data.x)
        y_pred = torch.nn.functional.softmax(h, dim=1)
        return y_pred

    def train_step(self, eval_=True):
        y_pred = self.get_prediction()
        y = self.data.y

        train_mask, val_mask, _ = self.data.get_masks()

        train_loss, train_acc = calc_metrics(y, y_pred, train_mask)
        train_loss.backward(retain_graph=False)
        if eval_:
            self.eval()
            val_loss, val_acc = calc_metrics(y, y_pred, val_mask)

            return train_loss, train_acc, val_loss, val_acc, 0, 0
        else:
            return train_loss, train_acc, torch.tensor(0), 0, 0, 0
