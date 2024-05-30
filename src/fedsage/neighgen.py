from operator import itemgetter
import logging
from itertools import compress


import torch
import numpy as np
from torch import optim
import torch.nn.functional as F
from torch_geometric.utils import subgraph

from src import *
from src.utils.graph import Graph
from src.fedsage.fedsage_models import MendGraph
from src.fedsage.fedsage_models import LocalSage_Plus
from src.fedsage.feature_loss import greedy_loss


class NeighGen:
    def __init__(
        self,
        id,
        x,
    ):
        self.num_pred = config.fedsage.num_pred
        self.id = id

        self.x = x

        self.original_graph: Graph = None
        self.impaired_graph: Graph = None
        self.mend_graph: Graph = None

    def prepare_data(self, graph: Graph):
        self.original_graph = graph
        (
            self.impaired_graph,
            self.true_missing,
            self.true_features,
            self.true_missing2,
            self.true_missing3,
        ) = self.create_impaired_graph(self.original_graph)

    def create_impaired_graph(self, graph: Graph):
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
                node_ids.to("cpu"),
                hide_length,
                replace=False,
            )
        )

        # node_mask = ~torch.isin(node_ids.to("cpu"), hide_nodes)
        node_mask = ~node_ids.cpu().unsqueeze(1).eq(hide_nodes).any(1)

        impaired_nodes = node_ids[node_mask]
        # impaired_edges0 = torch.isin(edges[0], impaired_nodes)
        # impaired_edges1 = torch.isin(edges[1], impaired_nodes)
        # edge_mask_ = impaired_edges0 & impaired_edges1
        # impaired_edges_ = edges[:, edge_mask_]

        impaired_edges, _, edge_mask = subgraph(
            impaired_nodes, edges, num_nodes=max(node_ids) + 1, return_edge_mask=True
        )
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

        other_edges = edges[:, ~edge_mask].cpu()
        true_missing = []
        missing_features = []
        cpu_nodes = impaired_nodes.cpu()
        for node_id in cpu_nodes:
            missing_neighbors = find_neighbors_(node_id, other_edges)
            if missing_neighbors.shape[0] > 0:
                if missing_neighbors.shape[0] > config.fedsage.num_pred:
                    missing_neighbors = missing_neighbors[: config.fedsage.num_pred]
                # node_idx = list(item_ge(graph.node_map, missing_neighbors))
                node_idx = np.array(
                    itemgetter(*missing_neighbors.numpy())(graph.node_map)
                )
                missing_features.append(self.original_graph.x[node_idx])
                true_missing.append(missing_neighbors.shape[0])
            else:
                missing_features.append([])
                true_missing.append(0)

        true_missing = torch.tensor(
            np.array(true_missing), dtype=torch.float32, device=device
        )

        edges2 = torch.hstack((graph.original_edge_index, graph.inter_edges))
        inter_degree = torch.bincount(edges2[0], minlength=max(node_ids) + 1)[node_ids]
        original_degree = torch.bincount(
            graph.original_edge_index[0], minlength=max(node_ids) + 1
        )[node_ids]
        true_missing3 = inter_degree - original_degree

        # original_degree = original_degree[node_mask]
        inter_degree = inter_degree[node_mask]
        impaired_degree = torch.bincount(
            impaired_graph.original_edge_index[0], minlength=max(node_ids) + 1
        )[impaired_graph.node_ids]
        # original_degree = degree(edges2[0], num_nodes=graph.num_nodes)
        # _, original_degree = edges2[0].unique(
        #     return_counts=True,
        # )
        # _, impaired_degree = impaired_graph.original_edge_index[0].unique(
        #     return_counts=True,
        # )
        # impaired_degree = degree(
        #     impaired_graph.original_edge_index[0], num_nodes=impaired_graph.num_nodes
        # )
        true_missing2 = inter_degree - impaired_degree

        return (
            impaired_graph,
            true_missing,
            missing_features,
            true_missing2,
            true_missing3,
        )

    def create_true_missing_features(original_graph: Graph, impaired_graph: Graph):
        true_missing = []
        true_features = []

        # node_ids = original_graph.node_ids
        edges = original_graph.edge_index.to("cpu")
        impaired_edges = impaired_graph.edge_index.to("cpu")

        for node_id in range(impaired_graph.num_nodes):
            subgraph_neighbors = find_neighbors_(
                node_id,
                edges,
                # include_external=config.fedsage.use_inter_connections,
            )
            impaired_graph_neighbors = find_neighbors_(node_id, impaired_edges)
            # missing_nodes = torch.tensor(
            #     np.setdiff1d(
            #         subgraph_neighbors, impaired_graph_neighbors, assume_unique=True
            #     )
            # )

            # mask = torch.isin(
            #     subgraph_neighbors,
            #     impaired_graph_neighbors,
            # )
            mask = subgraph_neighbors.unsqueeze(1).eq(impaired_graph_neighbors).any(1)

            missing_nodes = subgraph_neighbors[~mask]

            num_missing_neighbors = missing_nodes.shape[0]

            if num_missing_neighbors > 0:
                if num_missing_neighbors <= config.fedsage.num_pred:
                    missing_x = original_graph.x[missing_nodes]
                else:
                    missing_x = original_graph.x[
                        missing_nodes[: config.fedsage.num_pred]
                    ]
                    num_missing_neighbors = config.fedsage.num_pred
            else:
                missing_x = []
            true_missing.append(num_missing_neighbors)
            true_features.append(missing_x)
        true_missing = torch.tensor(
            np.array(true_missing),
            dtype=torch.float32,
            device=device,
        )
        # self.true_features = torch.tensor(np.array(self.true_features))

        return true_missing, true_features

    def set_model(self):
        self.predictor = LocalSage_Plus(
            feat_shape=self.impaired_graph.num_features,
            node_len=self.impaired_graph.num_nodes,
            n_classes=self.original_graph.num_classes,
            node_ids=self.impaired_graph.node_ids,
        )
        self.predictor.to(device)

        self.optimizer = optim.Adam(
            self.predictor.parameters(),
            lr=config.fedsage.neighgen_lr,
            weight_decay=config.model.weight_decay,
        )

    @torch.no_grad()
    def create_mend_graph(self, predict=True):
        self.predictor.eval()
        _, pred_features = self.predictor.predict_features(
            self.original_graph.x,
            self.original_graph.edge_index,
            self.true_missing3,
            predict=predict,
        )
        self.mend_graph = MendGraph.fill_graph(
            self.original_graph,
            pred_features,
        )

    def get_mend_graph(self):
        return self.mend_graph

    def predict_missing_neigh(self, predict=True):
        return self.predictor(
            self.impaired_graph.x,
            self.impaired_graph.edge_index,
            self.true_missing2,
            predict=predict,
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
        predict=True,
    ):
        loss_label = F.cross_entropy(y_pred[mask], y[mask])
        if predict:
            loss_missing = F.smooth_l1_loss(pred_missing[mask], true_missing[mask])
        else:
            loss_missing = torch.tensor(0, dtype=torch.float32, device=dev)

        # loss_feat = greedy_loss(pred_feat, true_feat, mask)
        masked_pred_feat = list(compress(pred_feat, mask))
        loss_feat = greedy_loss(masked_pred_feat, compress(true_feat, mask))
        if loss_feat is None:
            loss_feat = torch.tensor(0, dtype=torch.float32, device=dev)

        loss_list = []
        for inter_features in inter_features_list:
            # inter_loss_client = greedy_loss(pred_feat, inter_features, mask)
            inter_loss_client = greedy_loss(masked_pred_feat, inter_features)
            if inter_loss_client is not None:
                loss_list.append(inter_loss_client)

        if len(loss_list) > 0:
            inter_loss = torch.mean(torch.stack(loss_list), dim=0)
        else:
            inter_loss = torch.tensor(0, dtype=torch.float32, device=dev)

        loss = (
            config.fedsage.a * loss_missing
            + config.fedsage.b * loss_feat
            + config.fedsage.b * inter_loss
            + config.fedsage.c * loss_label
        )

        return loss

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

    def train_step(self, inter_client_features_creators, predict=True):
        pred_missing, pred_feat, y_pred = self.predict_missing_neigh(predict=predict)

        y = self.impaired_graph.y
        train_mask, val_mask, _ = self.impaired_graph.get_masks()

        inter_features = NeighGen.create_inter_features(
            inter_client_features_creators,
            self.impaired_graph.node_ids[train_mask],
        )

        train_loss = NeighGen.calc_loss(
            y,
            self.true_missing,
            self.true_features,
            y_pred,
            pred_missing,
            pred_feat,
            train_mask,
            inter_features,
            predict=predict,
        )

        train_loss.backward()

        # self.predictor.eval()
        # pred_missing, pred_feat, pred_label = self.predict_missing_neigh()
        (
            val_acc_label,
            val_acc_missing,
        ) = NeighGen.calc_accuracies(
            y,
            self.true_missing,
            y_pred,
            pred_missing,
            val_mask,
        )

        return train_loss.item(), val_acc_label, val_acc_missing
