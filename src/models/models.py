import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from src.models.GNN_models import GNN

from src.utils.config_parser import Config
from src.utils.graph import Graph

config = Config()
config.num_pred = 5
config.latent_dim = 128
config.hidden_layer_sizes = [64, 32]


def calc_accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()


@torch.no_grad()
def test(model, data: Graph):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    out = model(data.x, data.edge_index)
    # out = out[: len(data.test_mask)]
    label = data.y[: len(data.test_mask)]
    acc = calc_accuracy(out.argmax(dim=1)[data.test_mask], label[data.test_mask])
    return acc


class GraphSAGE(torch.nn.Module):
    """GraphSAGE"""

    def __init__(self, dim_in, dim_h, dim_out, dropout=0.5, last_layer="softmax"):
        super().__init__()
        self.layers = nn.ParameterList()
        input_layer = SAGEConv(dim_in, dim_h[0])
        self.layers.append(input_layer)
        for layer_num in range(len(dim_h) - 1):
            hidden_layer = SAGEConv(dim_h[layer_num], dim_h[layer_num + 1])
            self.layers.append(hidden_layer)

        # output_layer = SAGEConv(dim_h[-1],dim_out)
        output_layer = nn.Linear(dim_h[-1], dim_out)
        self.layers.append(output_layer)

        self.dropout = dropout
        self.last_layer = last_layer

    def reset_parameters(self) -> None:
        for layers in self.layers:
            layers.reset_parameters()

    def state_dict(self):
        weights = {}
        for id, layer in enumerate(self.layers):
            weights[f"layer{id}"] = layer.state_dict()
        return weights

    def load_state_dict(self, weights: dict) -> None:
        for id, layer in enumerate(self.layers):
            layer.load_state_dict(weights[f"layer{id}"])

    def forward(self, x, edge_index):
        h = x
        for layer in self.layers[:-1]:
            h = layer(h, edge_index).relu()
            h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.layers[-1](h)
        # out = self.layers[-1](h, edge_index)

        if self.last_layer == "softmax":
            return F.softmax(out, dim=1)
        elif self.last_layer == "relu":
            return F.relu(out)
        else:
            return out


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

    @torch.no_grad()
    def mend_graph(
        x,
        edges,
        predict_missing_nodes,
        predicted_features,
        node_ids=None,
    ):
        x = x.tolist()
        edges = edges.tolist()
        if node_ids is None:
            node_ids = list(range(len(x)))
        else:
            node_ids = node_ids.tolist()

        max_node_id = max(node_ids) + 1
        num_node = len(x)
        predict_missing_nodes = torch.round(predict_missing_nodes).int()
        predict_missing_nodes = torch.clip(
            predict_missing_nodes, 0, config.num_pred
        ).tolist()
        predicted_features = predicted_features.view(num_node, config.num_pred, -1)
        predicted_features = predicted_features.tolist()
        new_added_nodes = 0

        for i in range(len(x)):
            for j in range(predict_missing_nodes[i]):
                new_node_id = max_node_id + new_added_nodes
                node_ids.append(new_node_id)
                edges[0] += [node_ids[i], new_node_id]
                edges[1] += [new_node_id, node_ids[i]]
                x.append(predicted_features[i][j])
                new_added_nodes += 1

        x = torch.tensor(np.array(x), dtype=torch.float32)
        edges = torch.tensor(np.array(edges))
        node_ids = torch.tensor(np.array(node_ids))
        return x, edges, node_ids, new_added_nodes

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


class Gen(nn.Module):
    def __init__(self, latent_dim, dropout, num_pred, feat_shape):
        super(Gen, self).__init__()
        self.num_pred = num_pred
        self.feat_shape = feat_shape
        self.sample = Sampling()

        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 2048)
        self.fc_flat = nn.Linear(2048, self.num_pred * self.feat_shape)

        self.dropout = dropout

    def reset_parameters(self) -> None:
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc_flat.reset_parameters()
        self.sample.reset_parameters()

    def state_dict(self):
        weights = {}
        weights["fc1"] = self.fc1.state_dict()
        weights["fc2"] = self.fc2.state_dict()
        weights["fc_flat"] = self.fc2.state_dict()
        weights["sample"] = self.sample.state_dict()
        return weights

    def load_state_dict(self, weights: dict) -> None:
        self.fc1.load_state_dict(weights["fc1"])
        self.fc2.load_state_dict(weights["fc2"])
        self.fc_flat.load_state_dict(weights["fc_flat"])
        self.sample.load_state_dict(weights["sample"])

    def forward(self, x) -> torch.Tensor:
        x = self.sample(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.tanh(self.fc_flat(x))
        return x


class RegModel(nn.Module):
    def __init__(self, latent_dim):
        super(RegModel, self).__init__()
        self.reg_1 = nn.Linear(latent_dim, 1)

    def reset_parameters(self) -> None:
        self.reg_1.reset_parameters()

    def state_dict(self):
        weights = {}
        weights["reg1"] = self.reg_1.state_dict()
        return weights

    def load_state_dict(self, weights: dict) -> None:
        self.reg_1.load_state_dict(weights["reg1"])

    def forward(self, x):
        # x = F.relu(self.reg_1(x))
        x = config.num_pred * F.sigmoid(self.reg_1(x))
        return x.squeeze(1)


class LocalSage_Plus(nn.Module):
    def __init__(self, feat_shape, node_len, n_classes, node_ids):
        super(LocalSage_Plus, self).__init__()

        layer_sizes = [feat_shape] + config.hidden_layer_sizes + [config.latent_dim]
        self.encoder_model = GNN(
            layer_sizes=layer_sizes,
            dropout=config.model.dropout,
            last_layer="relu",
            # normalization="batch",
        )

        # self.encoder_model = GraphSAGE(
        #     dim_in=feat_shape,
        #     dim_h=config.hidden_layer_sizes,
        #     dim_out=config.latent_dim,
        #     dropout=config.model.dropout,
        #     last_layer="relu",
        # )

        self.reg_model = RegModel(latent_dim=config.latent_dim)

        self.gen = Gen(
            latent_dim=config.latent_dim,
            dropout=config.model.dropout,
            num_pred=config.num_pred,
            feat_shape=feat_shape,
        )

        self.mend_graph = MendGraph(
            node_len=node_len,
            num_pred=config.num_pred,
            feat_shape=feat_shape,
            node_ids=node_ids,
        )

        layer_sizes = [feat_shape] + config.hidden_layer_sizes + [n_classes]
        self.classifier = GNN(
            layer_sizes=layer_sizes,
            dropout=config.model.dropout,
            last_layer="softmax",
            normalization="batch",
        )

        # self.classifier = GraphSAGE(
        #     dim_in=feat_shape,
        #     dim_h=config.hidden_layer_sizes,
        #     dim_out=n_classes,
        #     dropout=config.model.dropout,
        #     last_layer="softmax",
        # )

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
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats, mend_edges, _, _ = MendGraph.mend_graph(
            feat, edges, degree, gen_feat
        )
        nc_pred = self.classifier(mend_feats, mend_edges)
        return degree, gen_feat, nc_pred[: feat.shape[0]]
