from ast import List
import logging

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.loader import NeighborLoader

from src.models.GNN_models import GraphSAGE, calc_accuracy, test
from src.utils.graph import Graph
from src.utils import config


class Classifier:
    def __init__(
        self,
        id,
        num_classes,
        logger=None,
    ):
        self.id = id
        self.num_classes = num_classes

        self.LOGGER = logger or logging

    def set_classifiers(self, dim_in=None):
        if dim_in is None:
            dim_in = self.graph.num_features

        self.GNN = GraphSAGE(
            dim_in=dim_in,
            dim_h=config.classifier_layer_sizes,
            dim_out=self.num_classes,
            dropout=config.dropout,
            last_layer="softmax",
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.GNN.parameters(), lr=config.lr, weight_decay=5e-4
        )

    def reindex_nodes(nodes, edges):
        index_map = -np.ones(max(nodes) + 1, dtype=np.int64)
        index_map[nodes] = np.arange(len(nodes))
        new_edges = np.vstack((index_map[edges[0, :]], index_map[edges[1, :]]))

        return index_map, new_edges

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

        self.train_loader = NeighborLoader(
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
                (
                    loss,
                    acc,
                    val_loss,
                    val_acc,
                ) = self.batch_training()
            else:
                (
                    loss,
                    acc,
                    val_loss,
                    val_acc,
                ) = self.whole_data_training()

            metrics = {
                "Train Loss": round(loss.item(), 4),
                "Train Acc": round(acc, 4),
                "Val Loss": round(val_loss.item(), 4),
                "Val Acc": round(val_acc, 4),
            }

            if bar:
                bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
                bar.set_postfix(metrics)
                bar.update(1)

            metrics["Epoch"] = epoch + 1
            res.append(metrics)

        if bar:
            self.LOGGER.info(f"{type} for client{self.id}:")
            self.LOGGER.info(metrics)

        if plot:
            Classifier.plot_results(res, self.id, type)

        return res

    def whole_data_training(self):
        self.optimizer.zero_grad()
        loss, acc, val_loss, val_acc = Classifier.train(
            self.graph.x,
            self.graph.y,
            self.graph.edge_index,
            self.GNN,
            self.criterion,
            self.graph.train_mask,
            self.graph.val_mask,
        )

        loss.backward()
        self.optimizer.step()

        return (
            loss,
            acc,
            val_loss,
            val_acc,
        )

    def batch_training(self):
        # Train on batches
        total_loss = torch.tensor([0], dtype=torch.float32, requires_grad=False)
        total_acc = 0
        total_val_loss = torch.tensor([0], dtype=torch.float32, requires_grad=False)
        total_val_acc = 0
        train_count = 0.000001
        val_count = 0.0000001
        for batch in self.train_loader:
            if batch.train_mask.any():
                self.optimizer.zero_grad()
                loss, acc, val_loss, val_acc = Classifier.train(
                    batch.x,
                    batch.y,
                    batch.edge_index,
                    self.GNN,
                    self.criterion,
                    batch.train_mask,
                    batch.val_mask,
                )
                train_count += 1

                total_loss += loss
                total_acc += acc
                loss.backward()
                self.optimizer.step()

            if batch.val_mask.any():
                val_count += 1
                total_val_loss += val_loss
                total_val_acc += val_acc

        return (
            total_loss / train_count,
            total_acc / train_count,
            total_val_loss / val_count,
            total_val_acc / val_count,
        )

    def train(
        x,
        y,
        edge_index,
        model: GraphSAGE,
        criterion,
        train_mask,
        val_mask,
    ):
        model.train()
        out = model(x, edge_index)

        # out = out[: len(data.train_mask)]
        # label = y
        # label = data.y[: len(data.train_mask)]

        train_loss = criterion(out[train_mask], y[train_mask])
        train_acc = calc_accuracy(
            out[train_mask].argmax(dim=1),
            y[train_mask],
        )

        # Validation
        # model.eval()
        with torch.no_grad():
            if val_mask.any():
                val_loss = criterion(out[val_mask], y[val_mask])
                val_acc = calc_accuracy(out[val_mask].argmax(dim=1), y[val_mask])
            else:
                val_loss = 0
                val_acc = 0

        return train_loss, train_acc, val_loss, val_acc

    def joint_train(
        self,
        structure_features,
        structure_loss,
        retain_graph=False,
    ):
        self.optimizer.zero_grad()
        x = torch.hstack((self.graph.x, structure_features))

        train_loss, train_acc, val_loss, val_acc = Classifier.train(
            x,
            self.graph.y,
            self.graph.edge_index,
            self.GNN,
            self.criterion,
            self.graph.train_mask,
            self.graph.val_mask,
        )

        total_train_loss = train_loss + structure_loss[self.graph.train_mask].mean()
        total_val_loss = val_loss + structure_loss[self.graph.val_mask].mean()

        total_train_loss.backward(retain_graph=retain_graph)
        self.optimizer.step()

        metrics = {
            "Train Loss": round(train_loss.item(), 4),
            "Train Acc": round(train_acc, 4),
            "Val Loss": round(val_loss.item(), 4),
            "Val Acc": round(val_acc, 4),
            "Total Train Loss": round(total_train_loss.item(), 4),
            "Total Val Loss": round(total_val_loss.item(), 4),
        }

        return metrics

    def calc_test_accuracy(self):
        return test(self.GNN, self.graph)

    def state_dict(self):
        return self.GNN.state_dict()

    def load_state_dict(self, weights):
        self.GNN.load_state_dict(weights)

    def reset_parameters(self):
        self.GNN.reset_parameters()

    def plot_results(res, client_id, type="local"):
        dataset = pd.DataFrame.from_dict(res)
        dataset.set_index("Epoch", inplace=True)

        loss_columns = list(filter(lambda x: x.endswith("Loss"), dataset.columns))
        dataset[loss_columns].plot()
        title = f"classifier loss client {client_id}"
        plt.title(title)
        plt.savefig(f"./{type} {title}.png")

        acc_columns = list(filter(lambda x: x.endswith("Acc"), dataset.columns))
        dataset[acc_columns].plot()
        title = f"classifier accuracy client {client_id}"
        plt.title(title)
        plt.savefig(f"./{type} {title}.png")
