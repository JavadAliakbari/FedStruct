import os
from itertools import compress
import logging
import time


import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.utils import subgraph

from src.utils.config_parser import Config
from src.utils.graph import Graph
from src.models.models import MendGraph
from src.models.models import calc_accuracy
from src.models.models import LocalSage_Plus
from src.models.feature_loss import greedy_loss
from src.utils.utils import *

path = os.environ.get("CONFIG_PATH")
config = Config(path)


class NeighGen:
    def __init__(
        self,
        id,
        num_classes,
        save_path="./",
        logger=None,
    ):
        self.num_pred = config.fedsage.num_pred
        self.id = id
        self.num_classes = num_classes
        self.save_path = save_path
        self.LOGGER = logger or logging

        self.original_graph: Graph = None
        self.impaired_graph: Graph = None
        self.mend_graph: Graph = None

    def prepare_data(self, graph: Graph):
        self.original_graph = graph
        self.impaired_graph = NeighGen.create_impaired_graph(self.original_graph)
        # self.create_true_missing_features(graph.x, graph.get_edges(), graph.node_ids)

        (
            self.true_missing,
            self.true_features,
        ) = NeighGen.create_true_missing_features(
            self.original_graph, self.impaired_graph
        )

    def create_impaired_graph(graph: Graph):
        node_ids = graph.node_ids
        edges = graph.get_edges()
        x = graph.x
        y = graph.y
        train_mask, val_mask, test_mask = graph.get_masks()

        train_portion = config.fedsage.impaired_train_nodes_ratio
        test_portion = config.fedsage.impaired_test_nodes_ratio
        hide_portion = (
            1 - train_portion - test_portion
        ) * config.fedsage.hidden_portion
        hide_length = int(len(node_ids) * hide_portion)

        hide_nodes = torch.tensor(
            np.random.choice(
                node_ids,
                hide_length,
                replace=False,
            )
        )

        node_mask = ~torch.isin(node_ids, hide_nodes)
        impaired_nodes = node_ids[node_mask]
        impaired_edges = subgraph(impaired_nodes, edges, num_nodes=max(node_ids) + 1)[0]
        impaired_x = x[node_mask]
        impaired_y = y[node_mask]
        impaired_train_mask = train_mask[node_mask]
        impaired_val_mask = val_mask[node_mask]
        impaired_test_mask = test_mask[node_mask]

        impaired_graph = Graph(
            x=impaired_x,
            y=impaired_y,
            edge_index=impaired_edges,
            node_ids=impaired_nodes,
            train_mask=impaired_train_mask,
            val_mask=impaired_val_mask,
            test_mask=impaired_test_mask,
        )

        # impaired_train_portion = train_portion * (1 - hide_portion)
        # impaired_test_portion = test_portion * (1 - hide_portion)

        # impaired_graph.add_masks(
        #     train_size=impaired_train_portion,
        #     test_size=impaired_test_portion,
        # )

        return impaired_graph

    def create_true_missing_features(original_graph: Graph, impaired_graph: Graph):
        true_missing = []
        true_features = []

        node_ids = original_graph.node_ids

        for node_id in impaired_graph.node_ids.numpy():
            subgraph_neighbors = original_graph.find_neigbors(
                node_id,
                include_external=config.fedsage.use_inter_connections,
            )
            impaired_graph_neighbors = impaired_graph.find_neigbors(node_id)
            missing_nodes = torch.tensor(
                np.setdiff1d(
                    subgraph_neighbors, impaired_graph_neighbors, assume_unique=True
                )
            )

            num_missing_neighbors = missing_nodes.shape[0]

            if num_missing_neighbors > 0:
                if num_missing_neighbors <= config.fedsage.num_pred:
                    missing_x = original_graph.x[torch.isin(node_ids, missing_nodes)]
                else:
                    missing_x = original_graph.x[
                        torch.isin(node_ids, missing_nodes[: config.fedsage.num_pred])
                    ]
            else:
                missing_x = []
            true_missing.append(num_missing_neighbors)
            true_features.append(missing_x)
        true_missing = torch.tensor(np.array(true_missing), dtype=torch.float32)
        # self.true_features = torch.tensor(np.array(self.true_features))

        return true_missing, true_features

    def set_model(self):
        self.predictor = LocalSage_Plus(
            feat_shape=self.impaired_graph.num_features,
            node_len=self.impaired_graph.num_nodes,
            n_classes=self.num_classes,
            node_ids=self.impaired_graph.node_ids,
        )

        self.optimizer = optim.Adam(
            self.predictor.parameters(),
            lr=config.model.lr,
            weight_decay=config.model.weight_decay,
        )

    @torch.no_grad()
    def create_mend_graph(self):
        pred_missing, pred_features, _ = NeighGen.predict_missing_neigh_(
            self.original_graph.x,
            self.original_graph.edge_index,
            self.predictor,
        )
        self.mend_graph = MendGraph.fill_graph(
            self.original_graph,
            pred_missing,
            pred_features,
        )

    def get_mend_graph(self):
        return self.mend_graph

    def predict_missing_neigh_(
        x,
        edge_index,
        predictor: LocalSage_Plus,
    ):
        num_nodes = x.shape[0]
        num_features = x.shape[1]
        pred_missing, pred_feat, pred_label = predictor(x, edge_index)

        pred_feat = pred_feat.view(
            num_nodes,
            -1,
            num_features,
        )
        # pred_missing = pred_missing.squeeze(1)

        return pred_missing, pred_feat, pred_label

    def predict_missing_neigh(self):
        return NeighGen.predict_missing_neigh_(
            self.impaired_graph.x,
            self.impaired_graph.edge_index,
            self.predictor,
        )

    def calc_loss(
        self,
        pred_missing,
        pred_feat,
        pred_label,
        mask,
        inter_client_features_creators=[],
    ):
        if mask == "train":
            mask = self.impaired_graph.train_mask
        elif mask == "val":
            mask = self.impaired_graph.val_mask

        loss_label = F.cross_entropy(
            pred_label[self.impaired_graph.train_mask],
            self.impaired_graph.y[self.impaired_graph.train_mask],
        )

        loss_missing = F.smooth_l1_loss(
            pred_missing[mask],
            self.true_missing[mask],
        )

        loss_feat = torch.tensor([0], dtype=torch.float32)
        inter_client_loss = torch.tensor([0], dtype=torch.float32)
        if loss_missing < 10000:
            true_train_features = list(compress(self.true_features, mask))

            loss_feat = greedy_loss(
                pred_feat[mask],
                true_train_features,
                pred_missing[mask],
            )

            if len(inter_client_features_creators) > 0:
                loss_list = torch.zeros(
                    len(inter_client_features_creators), dtype=torch.float32
                )
                for ind, inter_client_features_creator in enumerate(
                    inter_client_features_creators
                ):
                    loss_list[ind] = inter_client_features_creator(
                        pred_missing,
                        pred_feat,
                        self.true_missing,
                        self.impaired_graph.train_mask,
                    )
                inter_client_loss = loss_list.mean()

        return loss_missing, loss_feat, loss_label, inter_client_loss

    def calc_accuracy(
        self,
        pred_label,
        pred_missing,
        mask,
    ):
        if mask == "train":
            mask = self.impaired_graph.train_mask
        elif mask == "val":
            mask = self.impaired_graph.val_mask

        acc_missing = calc_accuracy(
            pred_missing[mask],
            self.true_missing[mask],
        )

        acc_label = calc_accuracy(
            torch.argmax(pred_label[mask], dim=1),
            self.impaired_graph.y[mask],
        )

        return acc_missing, acc_label

    def train_neighgen(
        self,
        optimizer,
        inter_client_features_creators: list = [],
    ):
        t = time.time()
        self.predictor.train()
        optimizer.zero_grad()

        pred_missing, pred_feat, pred_label = self.predict_missing_neigh()

        (
            loss_train_missing,
            loss_train_feat,
            loss_train_label,
            loss_train_client,
        ) = self.calc_loss(
            pred_missing,
            pred_feat,
            pred_label,
            "train",
            inter_client_features_creators,
        )

        loss = (
            config.fedsage.a * loss_train_missing
            + config.fedsage.b * loss_train_feat
            + config.fedsage.b * loss_train_client
            + config.fedsage.c * loss_train_label
        ).float()

        loss.backward()
        optimizer.step()

        acc_train_missing, acc_train_label = self.calc_accuracy(
            pred_label,
            # pred_missing.int(),
            torch.round(pred_missing).int(),
            mask="train",
        )

        self.predictor.eval()
        with torch.no_grad():
            val_missing, val_feat, val_label = self.predict_missing_neigh()

            loss_val_missing, loss_val_feat, loss_val_label, _ = self.calc_loss(
                val_missing,
                val_feat,
                val_label,
                mask="val",
            )

            acc_val_missing, acc_val_label = self.calc_accuracy(
                val_label,
                # val_missing.int(),
                torch.round(val_missing).int(),
                mask="val",
            )
            spend_time = time.time() - t

            return {
                "amt": round(acc_train_missing, 4),  # "acc_missing_train"
                "act": round(acc_train_label, 4),  # "acc_classification_train"
                "amv": round(acc_val_missing, 4),  # "acc_missing_val"
                "acv": round(acc_val_label, 4),  # "acc_classification_val"
                "lt": round(loss.item(), 4),  # "loss_train"
                "lmt": round(loss_train_missing.item(), 4),  # "loss_missing_train"
                "lct": round(loss_train_label.item(), 4),  # "loss_classification_train"
                "lft": round(loss_train_feat.item(), 4),  # "loss_feature_train"
                "lmv": round(loss_val_missing.item(), 4),  # "loss_missing_val"
                "lcv": round(loss_val_label.item(), 4),  # "loss_classification_val"
                "lfv": round(loss_val_feat.item(), 4),  # "loss_feature_val"
                "time": round(spend_time, 4),  # "time"
            }

    def fit(
        self,
        epochs,
        inter_client_features_creators: list = [],
        log=True,
        plot=True,
    ):
        if log:
            self.LOGGER.info(f"Start Neighgen Train for client {self.id}")
        res = []
        if log:
            bar = tqdm(total=epochs, bar_format="{l_bar}{bar:25}{r_bar}{bar:-10b}")
        for epoch in range(epochs):
            metrics = self.train_neighgen(
                self.optimizer,
                inter_client_features_creators,
            )

            metrics["Epoch"] = epoch + 1

            # self.LOGGER.info(metrics)

            res.append(metrics)

            if log:
                bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
                bar.set_postfix(metrics)
                bar.update(1)

        # self.trained = True
        if len(inter_client_features_creators) > 0:
            type = "fed"
        else:
            type = "local"

        if log:
            self.LOGGER.info(f"{type} for client{self.id}:")
            self.LOGGER.info(metrics)
        if plot:
            dataset = pd.DataFrame.from_dict(res)
            dataset.set_index("Epoch", inplace=True)
            # plt.figure()
            dataset[
                [
                    "lt",
                    "lmt",
                    "lct",
                    # "lft",
                    "lmv",
                    "lcv",
                    # "lfv",
                ]
            ].plot()
            title = f"Neighgen loss client {self.id}"
            plt.title(title)
            plt.savefig(f"{self.save_path}{type} {title}.png")

            dataset[
                [
                    "lft",
                    "lfv",
                ]
            ].plot()
            title = f"Neighgen Feature loss client {self.id}"
            plt.title(title)
            plt.savefig(f"{self.save_path}{type} {title}.png")

            dataset[
                [
                    "amt",
                    "act",
                    "amv",
                    "acv",
                ]
            ].plot()
            title = f"Neighgen accuracy Client {self.id}"
            plt.title(title)
            plt.savefig(f"{self.save_path}{type} {title}.png")

    def state_dict(self):
        return self.predictor.state_dict()

    def load_state_dict(self, weights):
        self.predictor.load_state_dict(weights)

    def reset_parameters(self):
        self.predictor.reset_parameters()
