from ast import List

import torch
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader

from src.utils.utils import *
from src.utils.graph import Graph
from src.classifier import Classifier
from src.utils.config_parser import Config
from src.models.GNN_models import GNNMLP, MPMLP, calc_accuracy, test, calc_f1_score

config = Config()


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

    def set_classifiers(self, dim_in=None, additional_layer_dims=0):
        if dim_in is None:
            dim_in = self.graph.num_features

        gnn_layer_sizes = [dim_in] + config.model.gnn_layer_sizes
        mlp_layer_sizes = (
            # [additional_layer_dims]
            [config.model.gnn_layer_sizes[-1] + additional_layer_dims]
            # + config.model.mlp_layer_sizes
            + [self.num_classes]
        )

        self.model: GNNMLP = GNNMLP(
            gnn_layer_sizes=gnn_layer_sizes,
            mlp_layer_sizes=mlp_layer_sizes,
            gnn_last_layer="linear",
            mlp_last_layer="softmax",
            dropout=config.model.dropout,
            batch_normalization=True,
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.model.lr,
            weight_decay=config.model.weight_decay,
        )

    def set_classifiers2(self, dim_in=None, additional_layer_dims=0):
        if dim_in is None:
            dim_in = self.graph.num_features

        mlp_layer_sizes = (
            [dim_in + additional_layer_dims]
            + config.model.mlp_layer_sizes
            + [self.num_classes]
        )

        self.model: MPMLP = MPMLP(
            num_gnn_layers=15,
            mlp_layer_sizes=mlp_layer_sizes,
            gnn_last_layer="linear",
            mlp_last_layer="softmax",
            dropout=config.model.dropout,
            batch_normalization=True,
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.model.lr,
            weight_decay=config.model.weight_decay,
        )

    def prepare_data(
        self,
        graph: Graph,
        batch_size: int = 16,
        num_neighbors: List = [5, 10],
    ):
        self.graph = graph
        if "train_mask" not in self.graph.keys:
            self.graph.add_masks()
        if self.graph.train_mask is None:
            self.graph.add_masks()

        self.data_loader = NeighborLoader(
            self.graph,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            # input_nodes=self.graph.train_mask,
            shuffle=True,
        )

    def get_feature_embeddings(self):
        x = self.graph.x
        edge_index = self.graph.edge_index
        h = self.model.gnn_step(x, edge_index)
        return h

    def predict_labels(self, x):
        h = self.model.mlp_step(x)
        return h

    def fit(
        self,
        epochs: int,
        batch=False,
        plot=False,
        bar=False,
        type="local",
    ):
        if bar:
            bar = tqdm(total=epochs, position=0)
        res = []
        if batch:
            data_loader = self.data_loader
        else:
            data_loader = [self.graph]
        for epoch in range(epochs):
            (
                loss,
                acc,
                TP,
                val_loss,
                val_acc,
                val_TP,
            ) = self.batch_training(data_loader)

            metrics = {
                "Train Loss": round(loss.item(), 4),
                "Train Acc": round(acc, 4),
                "Train F1 Score": round(TP, 4),
                "Val Loss": round(val_loss.item(), 4),
                "Val Acc": round(val_acc, 4),
                "Val F1 Score": round(val_TP, 4),
            }

            if bar:
                bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
                bar.set_postfix(metrics)
                bar.update(1)

            metrics["Epoch"] = epoch + 1
            res.append(metrics)

        if bar:
            self.LOGGER.info(f"{type} classifier for client{self.id}:")
            self.LOGGER.info(metrics)

        if plot:
            title = f"Client {self.id} {type} GNN"
            plot_metrics(res, title=title, save_path=self.save_path)

        return res

    def batch_training(self, data_loader):
        # Train on batches
        total_loss = 0
        total_acc = 0
        total_TP = 0
        total_val_loss = 0
        total_val_acc = 0
        total_val_TP = 0
        train_count = 1e-6
        val_count = 1e-6
        for batch in data_loader:
            if batch.train_mask.any():
                self.optimizer.zero_grad()
                loss, acc, TP, val_loss, val_acc, val_TP = GNNClassifier.train(
                    batch.x,
                    batch.y,
                    batch.edge_index,
                    self.model,
                    self.criterion,
                    batch.train_mask,
                    batch.val_mask,
                )
                train_count += 1

                total_loss += loss
                total_acc += acc
                total_TP += TP
                loss.backward()
                self.optimizer.step()

            if batch.val_mask.any():
                val_count += 1
                total_val_loss += val_loss
                total_val_acc += val_acc
                total_val_TP += val_TP

        return (
            total_loss / train_count,
            total_acc / train_count,
            total_TP / train_count,
            total_val_loss / val_count,
            total_val_acc / val_count,
            total_val_TP / val_count,
        )

    def train(
        x,
        y,
        edge_index,
        model: GNNMLP,
        criterion,
        train_mask,
        val_mask,
    ):
        model.train()
        out = model(x, edge_index)

        train_loss = criterion(out[train_mask], y[train_mask])
        train_acc = calc_accuracy(
            out[train_mask].argmax(dim=1),
            y[train_mask],
        )

        train_f1_score = calc_f1_score(out[train_mask].argmax(dim=1), y[train_mask])

        # Validation
        model.eval()
        with torch.no_grad():
            if val_mask.any():
                val_loss = criterion(out[val_mask], y[val_mask])
                val_acc = calc_accuracy(out[val_mask].argmax(dim=1), y[val_mask])
                val_f1_score = calc_f1_score(out[val_mask].argmax(dim=1), y[val_mask])
            else:
                val_loss = 0
                val_acc = 0
                val_f1_score = 0

        return train_loss, train_acc, train_f1_score, val_loss, val_acc, val_f1_score

    @torch.no_grad()
    def calc_test_accuracy(self):
        self.model.eval()
        return test(self.model, self.graph)
