from ast import List

import torch
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader

from src.models.GNN_models import GNN, calc_accuracy, test, calc_f1_score
from src.classifier import Classifier
from src.utils.graph import Graph
from src.utils.config_parser import Config

config = Config()


class GNNClassifier(Classifier):
    def __init__(
        self,
        id,
        num_classes,
        logger=None,
    ):
        super().__init__(
            id=id,
            num_classes=num_classes,
            logger=logger,
        )

    def set_classifiers(self, dim_in=None):
        if dim_in is None:
            dim_in = self.graph.num_features

        in_dims = [dim_in] + config.model.classifier_layer_sizes + [self.num_classes]
        self.model = GNN(
            in_dims=in_dims,
            dropout=config.model.dropout,
            linear_layer=True,
            last_layer="softmax",
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.model.lr, weight_decay=5e-4
        )

    def prepare_data(
        self,
        graph: Graph,
        batch_size: int = 16,
        num_neighbors: List = [5, 10],
        shuffle=True,
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
            shuffle=shuffle,
        )

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
        for epoch in range(epochs):
            if batch:
                data_loader = self.data_loader
            else:
                data_loader = [self.graph]

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
            Classifier.plot_results(res, self.id, type)

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
        model: GNN,
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

        return train_loss, train_acc, train_f1_score, val_loss, val_acc, val_f1_score

    def calc_test_accuracy(self):
        return test(self.model, self.graph)
