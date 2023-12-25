import os

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils.utils import *
from src.utils.graph import Data_
from src.classifier import Classifier
from src.utils.config_parser import Config
from src.models.GNN_models import MLP, calc_accuracy, calc_f1_score

path = os.environ.get("CONFIG_PATH")
config = Config(path)


class MLPClassifier(Classifier):
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

    def set_classifiers(self, dim_in=None):
        if dim_in is None:
            dim_in = self.data.num_features

        layer_sizes = (
            [dim_in] + config.feature_model.mlp_layer_sizes + [self.num_classes]
        )
        self.feature_model = MLP(
            layer_sizes=layer_sizes,
            last_layer="linear",
            dropout=config.model.dropout,
            normalization="layer",
            # normalization="batch",
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.feature_model.parameters(),
            lr=config.model.lr,
            weight_decay=config.model.weight_decay,
        )

    def prepare_data(
        self,
        data: Data_,
        batch_size: int = 16,
    ):
        self.data = data
        if "train_mask" not in self.data.keys:
            self.data.add_masks()
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

    def fit(
        self,
        epochs: int,
        batch=False,
        plot=False,
        log=False,
        type="local",
    ):
        if log:
            bar = tqdm(total=epochs, position=0)

        if batch:
            train_loader = self.train_loader
            val_loader = self.val_loader
        else:
            train_x = self.data.x[self.data.train_mask]
            train_y = self.data.y[self.data.train_mask]
            val_x = self.data.x[self.data.val_mask]
            val_y = self.data.y[self.data.val_mask]
            train_loader = [(train_x, train_y)]
            val_loader = [(val_x, val_y)]
        res = []
        for epoch in range(epochs):
            (
                loss,
                acc,
                f1_score,
                val_loss,
                val_acc,
                val_f1_score,
            ) = self.batch_training(train_loader, val_loader)

            metrics = {
                "Train Loss": round(loss.item(), 4),
                "Train Acc": round(acc, 4),
                "Train F1 Score": round(f1_score, 4),
                "Val Loss": round(val_loss.item(), 4),
                "Val Acc": round(val_acc, 4),
                "Val F1 Score": round(val_f1_score, 4),
            }

            if log:
                bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
                bar.set_postfix(metrics)
                bar.update(1)

            metrics["Epoch"] = epoch + 1
            res.append(metrics)

        if log:
            self.LOGGER.info(f"{type} classifier for client{self.id}:")
            self.LOGGER.info(metrics)

        if plot:
            title = f"Client {self.id} {type} MLP"
            plot_metrics(res, title=title, save_path=self.save_path)

        return res

    def batch_training(self, train_loader, val_loader):
        # Train on batches
        total_loss = torch.tensor([0], dtype=torch.float32, requires_grad=False)
        total_acc = 0
        total_f1_score = 0
        total_val_loss = torch.tensor([0], dtype=torch.float32, requires_grad=False)
        total_val_acc = 0
        total_val_f1_score = 0
        train_count = 1e-6
        val_count = 1e-6
        self.feature_model.train()
        for train_x, train_y in train_loader:
            self.optimizer.zero_grad()

            out = self.feature_model(train_x)

            train_loss = self.criterion(out, train_y)
            train_acc = calc_accuracy(out.argmax(dim=1), train_y)

            train_f1_score = calc_f1_score(out.argmax(dim=1), train_y)

            total_loss += train_loss
            total_acc += train_acc
            total_f1_score += train_f1_score
            train_count += 1

            train_loss.backward()
            self.optimizer.step()

        self.feature_model.eval()
        with torch.no_grad():
            for val_x, val_y in val_loader:
                out = self.feature_model(val_x)
                val_loss = self.criterion(out, val_y)
                val_acc = calc_accuracy(out.argmax(dim=1), val_y)
                val_f1_score = calc_f1_score(out.argmax(dim=1), val_y)

                total_val_loss += val_loss
                total_val_acc += val_acc
                total_val_f1_score += val_f1_score
                val_count += 1

        return (
            total_loss / train_count,
            total_acc / train_count,
            total_f1_score / train_count,
            total_val_loss / val_count,
            total_val_acc / val_count,
            total_val_f1_score / val_count,
        )

    def step(
        x,
        y,
        model: MLP,
        criterion,
        train_mask,
        val_mask,
    ):
        # model.train()
        out = model(x)

        train_loss = criterion(out[train_mask], y[train_mask])
        train_acc = calc_accuracy(
            out[train_mask].argmax(dim=1),
            y[train_mask],
        )

        train_TP = calc_f1_score(out[train_mask].argmax(dim=1), y[train_mask])

        # Validation
        # model.eval()
        with torch.no_grad():
            if val_mask.any():
                val_loss = criterion(out[val_mask], y[val_mask])
                val_acc = calc_accuracy(out[val_mask].argmax(dim=1), y[val_mask])
                val_TP = calc_f1_score(out[val_mask].argmax(dim=1), y[val_mask])
            else:
                val_loss = 0
                val_acc = 0

        return train_loss, train_acc, train_TP, val_loss, val_acc, val_TP

    @torch.no_grad()
    def calc_test_accuracy(self, metric="acc"):
        self.feature_model.eval()
        test_x, test_y = self.test_data
        out = self.feature_model(test_x)
        if metric == "acc":
            val_acc = calc_accuracy(out.argmax(dim=1), test_y)
        elif metric == "f1":
            val_acc = calc_f1_score(out.argmax(dim=1), test_y)

        return val_acc

    def get_prediction(self):
        h = self.feature_model(self.data.x)
        y_pred = torch.nn.functional.softmax(h, dim=1)
        return y_pred

    def train_step(self, scale=False):
        y_pred = self.get_prediction()
        y = self.data.y

        train_mask, val_mask, _ = self.data.get_masks()

        train_loss, train_acc, train_f1_score = calc_metrics(
            y, y_pred, train_mask, self.criterion
        )
        val_loss, val_acc, val_f1_score = calc_metrics(
            y, y_pred, val_mask, self.criterion
        )

        if scale:
            train_loss *= self.data.num_nodes
        train_loss.backward(retain_graph=True)

        return train_loss, train_acc, train_f1_score, val_loss, val_acc, val_f1_score
