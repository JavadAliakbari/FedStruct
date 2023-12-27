import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from src.models.GNN_models import GNN, MLP
from src.utils.config_parser import Config
from src.utils.graph import Graph

path = os.environ.get("CONFIG_PATH")
config = Config(path)


class Sampling(nn.Module):
    def __init__(self):
        super(Sampling, self).__init__()

    def reset_parameters(self) -> None:
        pass

    def state_dict(self):
        weights = {}
        return weights

    def load_state_dict(self, weights: dict) -> None:
        pass

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rand = torch.normal(0, 1, size=inputs.shape)
        # if config.cuda:
        #     return inputs + rand.cuda()
        # else:
        return inputs + rand


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def reset_parameters(self) -> None:
        pass

    def state_dict(self):
        weights = {}
        return weights

    def load_state_dict(self, weights: dict) -> None:
        pass

    def forward(self, x):
        return x.view(self.shape)


class MendGraph(nn.Module):
    def __init__(self, node_len, num_pred, feat_shape, node_ids):
        super(MendGraph, self).__init__()
        self.num_pred = num_pred
        self.feat_shape = feat_shape
        self.org_node_len = node_len
        self.node_len = self.org_node_len * (self.num_pred + 1)
        self.node_ids = node_ids
        for param in self.parameters():
            param.requires_grad = False

    def reset_parameters(self) -> None:
        pass

    def state_dict(self):
        weights = {}
        return weights

    def load_state_dict(self, weights: dict) -> None:
        pass

    # @torch.no_grad()
    # def mend_graph(
    #     x,
    #     edges,
    #     predict_missing_nodes,
    #     predicted_features,
    #     node_ids=None,
    # ):
    #     x = x.tolist()
    #     edges = edges.tolist()
    #     if node_ids is None:
    #         node_ids = list(range(len(x)))
    #     else:
    #         node_ids = node_ids.tolist()

    #     max_node_id = max(node_ids) + 1
    #     num_node = len(x)
    #     predict_missing_nodes = torch.round(predict_missing_nodes).int()
    #     predict_missing_nodes = torch.clip(
    #         predict_missing_nodes, 0, config.fedsage.num_pred
    #     ).tolist()
    #     predicted_features = predicted_features.view(
    #         num_node,
    #         config.fedsage.num_pred,
    #         -1,
    #     )
    #     predicted_features = predicted_features.tolist()
    #     new_added_nodes = 0

    #     new_x = []
    #     for i in range(len(x)):
    #         for j in range(predict_missing_nodes[i]):
    #             new_node_id = max_node_id + new_added_nodes
    #             node_ids.append(new_node_id)
    #             edges[0] += [node_ids[i], new_node_id]
    #             edges[1] += [new_node_id, node_ids[i]]
    #             x.append(predicted_features[i][j])
    #             new_added_nodes += 1

    #     # all_x = torch.cat((x, new_x))
    #     x = torch.tensor(np.array(x), dtype=torch.float32)
    #     # concatenated_x = torch.cat([x, *new_x], dim=0)
    #     edges = torch.tensor(np.array(edges))
    #     node_ids = torch.tensor(np.array(node_ids))
    #     return x, edges, node_ids, new_added_nodes

    def mend_graph(
        x,
        edges,
        predict_missing_nodes,
        predicted_features,
        node_ids=None,
    ):
        # x = x.tolist()
        edges = edges.tolist()
        if node_ids is None:
            node_ids = list(range(len(x)))
        else:
            node_ids = node_ids.tolist()

        max_node_id = max(node_ids) + 1
        num_node = x.shape[0]
        predict_missing_nodes = torch.round(predict_missing_nodes).int()
        predict_missing_nodes = torch.clip(
            predict_missing_nodes, 0, config.fedsage.num_pred
        ).tolist()
        predicted_features = predicted_features.view(
            num_node,
            config.fedsage.num_pred,
            -1,
        )
        # predicted_features = predicted_features.tolist()
        new_added_nodes = 0

        new_x = []
        for i in range(x.shape[0]):
            for j in range(predict_missing_nodes[i]):
                new_node_id = max_node_id + new_added_nodes
                node_ids.append(new_node_id)
                edges[0] += [node_ids[i], new_node_id]
                edges[1] += [new_node_id, node_ids[i]]
                new_x.append(predicted_features[i, j].unsqueeze(0))
                new_added_nodes += 1

        # all_x = torch.cat((x, new_x))
        concatenated_x = torch.cat([x, *new_x], dim=0)
        edges = torch.tensor(np.array(edges))
        node_ids = torch.tensor(np.array(node_ids))
        return concatenated_x, edges, node_ids, new_added_nodes

    @torch.no_grad()
    def fill_graph(
        graph: Graph,
        predict_missing_nodes,
        predicted_features,
    ):
        y = graph.y
        train_mask = graph.train_mask
        test_mask = graph.test_mask
        val_mask = graph.val_mask

        x, edges, node_ids, new_added_nodes = MendGraph.mend_graph(
            graph.x,
            graph.get_edges(),
            predict_missing_nodes,
            predicted_features,
            graph.node_ids,
        )

        train_mask = torch.hstack(
            (train_mask, torch.zeros(new_added_nodes, dtype=torch.bool))
        )
        test_mask = torch.hstack(
            (test_mask, torch.zeros(new_added_nodes, dtype=torch.bool))
        )
        val_mask = torch.hstack(
            (val_mask, torch.zeros(new_added_nodes, dtype=torch.bool))
        )

        y_shape = list(y.shape)
        y_shape[0] = new_added_nodes
        y = torch.hstack((y, torch.zeros(y_shape, dtype=y.dtype)))

        mend_graph = Graph(
            x=x,
            y=y,
            edge_index=edges,
            node_ids=node_ids,
            train_mask=train_mask,
            test_mask=test_mask,
            val_mask=val_mask,
        )

        return mend_graph

    def forward(self, org_feats, org_edges, pred_missing, gen_feats):
        fill_edges, fill_feats = self.mend_graph(
            org_feats, org_edges, pred_missing, gen_feats
        )
        return fill_feats, fill_edges


class Gen(MLP):
    # def __init__(self, latent_dim, dropout, num_pred, feat_shape):
    def __init__(
        self,
        layer_sizes,
        last_layer="softmax",
        dropout=0.5,
        normalization=None,
    ):
        super().__init__(
            layer_sizes=layer_sizes,
            last_layer=last_layer,
            dropout=dropout,
            normalization=normalization,
        )
        self.sample = Sampling()

    def forward(self, x) -> torch.Tensor:
        x = self.sample(x)
        x = super().forward(x)
        return x


class LocalSage_Plus(nn.Module):
    def __init__(self, feat_shape, node_len, n_classes, node_ids):
        super(LocalSage_Plus, self).__init__()

        layer_sizes = (
            [feat_shape]
            + config.fedsage.hidden_layer_sizes
            + [config.fedsage.latent_dim]
        )
        self.encoder_model = GNN(
            layer_sizes=layer_sizes,
            dropout=config.model.dropout,
            last_layer="relu",
            # normalization="batch",
        )

        self.reg_model = MLP(
            layer_sizes=[config.fedsage.latent_dim, 1],
            dropout=config.model.dropout,
            # last_layer="softmax",
            last_layer="relu",
            # normalization="batch",
        )

        gen_layer_sizes = [
            config.fedsage.latent_dim,
            config.fedsage.neighen_feature_gen[0],
            config.fedsage.neighen_feature_gen[1],
            config.fedsage.num_pred * feat_shape,
        ]
        self.gen = Gen(
            layer_sizes=gen_layer_sizes,
            last_layer="tanh",
            dropout=config.model.dropout,
            # normalization="batch",
        )

        self.mend_graph = MendGraph(
            node_len=node_len,
            num_pred=config.fedsage.num_pred,
            feat_shape=feat_shape,
            node_ids=node_ids,
        )

        layer_sizes = [feat_shape] + config.fedsage.hidden_layer_sizes + [n_classes]
        self.classifier = GNN(
            layer_sizes=layer_sizes,
            dropout=config.model.dropout,
            last_layer="softmax",
            normalization="batch",
        )

    def reset_parameters(self) -> None:
        self.encoder_model.reset_parameters()
        self.reg_model.reset_parameters()
        self.gen.reset_parameters()
        self.mend_graph.reset_parameters()
        self.classifier.reset_parameters()

    def state_dict(self):
        weights = {}
        weights["encoder"] = self.encoder_model.state_dict()
        weights["reg"] = self.reg_model.state_dict()
        weights["gen"] = self.gen.state_dict()
        weights["mend"] = self.mend_graph.state_dict()
        weights["classifier"] = self.classifier.state_dict()
        return weights

    def load_state_dict(self, weights: dict) -> None:
        self.encoder_model.load_state_dict(weights["encoder"])
        self.reg_model.load_state_dict(weights["reg"])
        self.gen.load_state_dict(weights["gen"])
        self.mend_graph.load_state_dict(weights["mend"])
        self.classifier.load_state_dict(weights["classifier"])

    def forward(self, feat, edges):
        x = self.encoder_model(feat, edges)
        # degree = config.fedsage.num_pred * self.reg_model(x).squeeze(1)
        degree = self.reg_model(x).squeeze(1)
        gen_feat = self.gen(x)
        mend_feats, mend_edges, _, _ = MendGraph.mend_graph(
            feat, edges, degree, gen_feat
        )
        nc_pred = self.classifier(mend_feats, mend_edges)
        return degree, gen_feat, nc_pred[: feat.shape[0]]
