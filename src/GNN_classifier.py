from ast import List

import torch
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader

from src.utils.utils import *
from src.utils.graph import Graph
from src.classifier import Classifier
from src.utils.config_parser import Config
from src.models.GNN_models import (
    ModelBinder,
    ModelSpecs,
    calc_accuracy,
    test,
    calc_f1_score,
)

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

    def set_GNN_classifier(self, dim_in=None, additional_layer_dims=0):
        if dim_in is None:
            dim_in = self.graph.num_features

        gnn_layer_sizes = [dim_in] + config.feature_model.gnn_layer_sizes
        mlp_layer_sizes = (
            [config.feature_model.gnn_layer_sizes[-1] + additional_layer_dims]
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
                final_activation_function="softmax",
                normalization=None,
            ),
        ]

        self.model: ModelBinder = ModelBinder(model_specs)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.model.lr,
            weight_decay=config.model.weight_decay,
        )

    def set_MP_classifier(self, dim_in=None, additional_layer_dims=0):
        if dim_in is None:
            dim_in = self.graph.num_features

        mlp_layer_sizes = [dim_in] + config.feature_model.mlp_layer_sizes
        decision_layer_sizes = [
            config.feature_model.mlp_layer_sizes[-1] + additional_layer_dims,
            self.num_classes,
        ]

        # gnn_layer_sizes = [dim_in] + config.feature_model.gnn_layer_sizes
        # decision_layer_sizes = [
        #     config.feature_model.gnn_layer_sizes[-1] + additional_layer_dims,
        #     self.num_classes,
        # ]

        model_specs = [
            ModelSpecs(
                type="MLP",
                layer_sizes=mlp_layer_sizes,
                # final_activation_function="softmax",
                final_activation_function="linear",
                # final_activation_function="relu",
                normalization="layer",
            ),
            ModelSpecs(
                type="MP",
                num_layers=config.feature_model.mp_layers,
            ),
            # ModelSpecs(
            #     type="GNN",
            #     layer_sizes=gnn_layer_sizes,
            #     final_activation_function="linear",
            #     normalization="batch",
            # ),
            ModelSpecs(
                type="MLP",
                layer_sizes=decision_layer_sizes,
                final_activation_function="softmax",
                # final_activation_function="linear",
                normalization=None,
            ),
        ]

        self.model: ModelBinder = ModelBinder(model_specs)

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
        h = self.graph.x
        edge_index = self.graph.edge_index
        for model in self.model[:-1]:
            h = self.model.step(model, h, edge_index)
        return h

    def predict_labels(self, h):
        edge_index = self.graph.edge_index
        h = self.model.step(self.model[-1], h, edge_index)
        return h

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
        res = []
        if batch:
            data_loader = self.data_loader
        else:
            data_loader = [self.graph]
        for epoch in range(epochs):
            (
                loss,
                acc,
                f1_score,
                val_loss,
                val_acc,
                val_f1_score,
            ) = self.batch_training(data_loader)

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
            title = f"Client {self.id} {type} GNN"
            plot_metrics(res, title=title, save_path=self.save_path)

        return res

    def batch_training(self, data_loader):
        # Train on batches
        total_loss = 0
        total_acc = 0
        total_f1_score = 0
        total_val_loss = 0
        total_val_acc = 0
        total_val_f1_score = 0
        train_count = 1e-6
        val_count = 1e-6
        self.model.train()
        for batch in data_loader:
            if batch.train_mask.any():
                self.optimizer.zero_grad()
                (
                    loss,
                    acc,
                    f1_score,
                    val_loss,
                    val_acc,
                    val_f1_score,
                ) = GNNClassifier.step(
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
                total_f1_score += f1_score
                loss.backward()
                self.optimizer.step()

            if batch.val_mask.any():
                val_count += 1
                total_val_loss += val_loss
                total_val_acc += val_acc
                total_val_f1_score += val_f1_score

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
        edge_index,
        model: ModelBinder,
        criterion,
        train_mask,
        val_mask,
    ):
        # model.train()
        out = model(x, edge_index)

        train_loss = criterion(out[train_mask], y[train_mask])
        train_acc = calc_accuracy(
            out[train_mask].argmax(dim=1),
            y[train_mask],
        )

        train_f1_score = calc_f1_score(out[train_mask].argmax(dim=1), y[train_mask])

        # Validation
        # model.eval()
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
    def calc_test_accuracy(self, metric="acc"):
        self.model.eval()
        out = self.model(self.graph.x, self.graph.edge_index)
        label = self.graph.y[self.graph.test_mask]
        predicted = out.argmax(dim=1)[self.graph.test_mask]
        if metric == "acc":
            val_acc = calc_accuracy(predicted, label)
        elif metric == "f1":
            val_acc = calc_f1_score(predicted, label)

        return val_acc
