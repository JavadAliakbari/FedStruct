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

from src.utils.utils import *
from src.utils.config_parser import Config
from src.utils.graph import Graph
from src.models.models import MendGraph
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
            lr=config.fedsage.neighgen_lr,
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

    def create_inter_features(
        inter_client_features_creators,
        mask,
    ):
        # return []
        inter_features = []
        for inter_client_features_creator in inter_client_features_creators:
            inter_feature_client = inter_client_features_creator(mask)

            inter_features.append(inter_feature_client)

        return inter_features

    def calc_loss(
        y,
        true_missing,
        true_feat,
        y_pred,
        pred_missing,
        pred_feat,
        mask,
        inter_features_list=[],
    ):
        loss_label = F.cross_entropy(y_pred[mask], y[mask])
        loss_missing = F.smooth_l1_loss(pred_missing[mask], true_missing[mask])

        loss_feat = greedy_loss(
            pred_feat[mask],
            true_feat,
            pred_missing[mask],
        )

        loss_list = []
        for inter_features in inter_features_list:
            inter_loss_client = greedy_loss(
                pred_feat[mask],
                inter_features,
                pred_missing[mask],
            )
            loss_list.append(inter_loss_client)

        if len(loss_list) > 0:
            inter_loss = torch.mean(torch.stack(loss_list), dim=0)
        else:
            inter_loss = torch.tensor([0], dtype=torch.float32)

        return loss_label, loss_missing, loss_feat, inter_loss

    @torch.no_grad()
    def calc_accuracies(
        y,
        true_missing,
        y_pred,
        pred_missing,
        mask,
    ):
        acc_missing = calc_accuracy(
            pred_missing[mask],
            true_missing[mask],
        )

        acc_label = calc_accuracy(
            torch.argmax(y_pred[mask], dim=1),
            y[mask],
        )

        return acc_label, acc_missing

    def calc_metrics(
        y,
        true_missing,
        true_feat,
        y_pred,
        pred_missing,
        pred_feat,
        mask,
        inter_features_list=[],
    ):
        loss_label, loss_missing, loss_feat, inter_loss = NeighGen.calc_loss(
            y,
            true_missing,
            true_feat,
            y_pred,
            pred_missing,
            pred_feat,
            mask,
            inter_features_list,
        )

        acc_label, acc_missing = NeighGen.calc_accuracies(
            y,
            true_missing,
            y_pred,
            pred_missing,
            mask,
        )

        return loss_label, loss_missing, loss_feat, inter_loss, acc_label, acc_missing

    @torch.no_grad()
    def calc_test_accuracy(self, metric="label"):
        self.predictor.eval()

        pred_missing, pred_feat, pred_label = self.predict_missing_neigh()
        y = self.impaired_graph.y
        test_mask = self.impaired_graph.test_mask

        acc_label, acc_missing = NeighGen.calc_accuracies(
            y, self.true_missing, pred_label, pred_missing, test_mask
        )

        if metric == "label":
            return acc_label
        else:
            return acc_missing

    def train(self, mode: bool = True):
        self.predictor.train(mode)

    def eval(self):
        self.predictor.eval()

    def state_dict(self):
        return self.predictor.state_dict()

    def load_state_dict(self, weights):
        self.predictor.load_state_dict(weights)

    def reset_parameters(self):
        self.predictor.reset_parameters()

    def update_model(self):
        self.optimizer.step()

    def reset_classifier(self):
        self.optimizer.zero_grad()

    def train_step(self, inter_client_features_creators):
        pred_missing, pred_feat, pred_label = self.predict_missing_neigh()

        y = self.impaired_graph.y
        train_mask, val_mask, _ = self.impaired_graph.get_masks()

        inter_features = NeighGen.create_inter_features(
            inter_client_features_creators,
            train_mask,
        )

        (
            train_loss_label,
            train_loss_missing,
            train_loss_feat,
            train_inter_loss,
            train_acc_label,
            train_acc_missing,
        ) = NeighGen.calc_metrics(
            y,
            self.true_missing,
            self.true_features,
            pred_label,
            pred_missing,
            pred_feat,
            train_mask,
            inter_features,
        )

        train_loss = (
            config.fedsage.a * train_loss_missing
            + config.fedsage.b * train_loss_feat
            + config.fedsage.b * train_inter_loss
            + config.fedsage.c * train_loss_label
        )

        train_loss.backward()

        # self.predictor.eval()
        # pred_missing, pred_feat, pred_label = self.predict_missing_neigh()

        (
            val_loss_label,
            val_loss_missing,
            val_loss_feat,
            val_inter_loss,
            val_acc_label,
            val_acc_missing,
        ) = NeighGen.calc_metrics(
            y,
            self.true_missing,
            self.true_features,
            pred_label,
            pred_missing,
            pred_feat,
            val_mask,
        )

        return train_loss, val_acc_label, val_acc_missing, val_loss_feat
