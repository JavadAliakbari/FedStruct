import torch
import numpy as np
import torch.nn as nn

from src import *
from src.models.model_binders import GNN, MLP
from src.utils.graph import Graph


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
        rand = torch.normal(0, 1, size=inputs.shape, device=device)
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
        predicted_features,
        node_ids=None,
    ):
        if node_ids is None:
            node_ids_ = torch.arange(len(x))
        else:
            node_ids_ = node_ids.cpu()

        max_node_id = max(node_ids_).cpu().item() + 1
        num_nodes = x.shape[0]
        new_edges0 = []
        new_edges1 = []

        new_node_id = max_node_id
        new_x = []
        for i in range(num_nodes):
            num_neighbors = len(predicted_features[i])
            if num_neighbors > 0:
                new_x.append(predicted_features[i])
                l0 = num_neighbors * [node_ids_[i].item()]
                l1 = list(range(new_node_id, new_node_id + num_neighbors))
                new_edges0 += l0 + l1
                new_edges1 += l1 + l0
                new_node_id += num_neighbors

        if new_node_id > max_node_id:
            # print(new_node_id - max_node_id)
            concatenated_x = torch.cat([x, *new_x], dim=0)
            new_edges = np.array([new_edges0, new_edges1], dtype=int)
            new_edges = torch.tensor(new_edges, device=device)
            edges = torch.hstack((edges, new_edges))
            return concatenated_x, edges
        else:
            return x, edges

    @torch.no_grad()
    def fill_graph(
        graph: Graph,
        # predict_missing_nodes,
        predicted_features,
    ):
        y = graph.y
        train_mask = graph.train_mask
        test_mask = graph.test_mask
        val_mask = graph.val_mask

        x, edges = MendGraph.mend_graph(
            graph.x,
            graph.get_edges(),
            predicted_features,
            graph.node_ids,
        )
        max_node_id = max(graph.node_ids).cpu().item() + 1
        new_added_nodes = x.shape[0] - graph.x.shape[0]
        new_node_id = max_node_id + new_added_nodes
        new_node_ids = torch.arange(
            max_node_id, new_node_id, device=graph.node_ids.device
        )
        node_ids = torch.hstack((graph.node_ids, new_node_ids))

        train_mask = torch.hstack(
            (train_mask, torch.zeros(new_added_nodes, dtype=torch.bool, device=dev))
        )
        test_mask = torch.hstack(
            (test_mask, torch.zeros(new_added_nodes, dtype=torch.bool, device=dev))
        )
        val_mask = torch.hstack(
            (val_mask, torch.zeros(new_added_nodes, dtype=torch.bool, device=dev))
        )

        y_shape = list(y.shape)
        y_shape[0] = new_added_nodes
        y = torch.hstack((y, torch.zeros(y_shape, dtype=y.dtype, device=y.device)))

        mend_graph = Graph(
            x=x,
            y=y,
            edge_index=edges,
            node_ids=node_ids,
            train_mask=train_mask,
            test_mask=test_mask,
            val_mask=val_mask,
            num_classes=graph.num_classes,
        )

        return mend_graph


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

    def forward(self, feat, edges, true_missing, predict=True):
        degree, gen_feat = self.predict_features(
            feat, edges, true_missing, predict=predict
        )
        mend_feats, mend_edges = MendGraph.mend_graph(feat, edges, gen_feat)
        # nc_pred = self.classifier(feat, edges)
        nc_pred = self.classifier(mend_feats, mend_edges)
        return degree, gen_feat, nc_pred[: feat.shape[0]]

    def predict_features(self, feat, edges, true_missing, predict=True):
        x = self.encoder_model(feat, edges)
        # degree = config.fedsage.num_pred * self.reg_model(x).squeeze(1)
        if predict:
            degree = self.reg_model(x).squeeze(1)
        else:
            degree = true_missing
        # degree = torch.ones(
        #     size=(1, feat.shape[0]), requires_grad=True, dtype=torch.float32
        # ).squeeze(0)
        gen_feat = self.gen(x)
        new_gen_feat = LocalSage_Plus.cut_feats(gen_feat, degree)

        return degree, new_gen_feat

    def cut_feats(pred_feat, pred_deg):
        with torch.no_grad():
            num_nodes = pred_deg.shape[0]
            pr = pred_deg.cpu()
            pr = torch.round(pr).int()
            pr = torch.clip(pr, 0, config.fedsage.num_pred)
        pf = pred_feat.view(num_nodes, config.fedsage.num_pred, -1)
        # num_features = int(round(pred_feat.shape[1] / config.fedsage.num_pred))

        new_x = []

        for i in range(num_nodes):
            num_neighbors = pr[i]
            if num_neighbors > 0:
                tensor = pf[i, :num_neighbors]
                # tensor = tensor.view(num_neighbors, -1)
            else:
                tensor = []
            new_x.append(tensor)

        return new_x
