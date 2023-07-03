import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv
import torch.nn.functional as F

from src.utils import config
from src.utils.graph import Graph


class GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN, self).__init__()
        self.sage1 = GCNConv(nfeat, nhid)
        # self.sage1 = SAGEConv(nfeat, nhid)
        self.sage2 = GCNConv(nhid, nclass)
        # self.sage2 = SAGEConv(nhid, nclass)
        self.dropout = dropout

    def reset_parameters(self) -> None:
        self.sage1.reset_parameters()
        self.sage2.reset_parameters()

    def state_dict(self):
        weights = {}
        weights["sage1"] = self.sage1.state_dict()
        weights["sage2"] = self.sage2.state_dict()
        return weights

    def load_state_dict(self, weights: dict) -> None:
        self.sage1.load_state_dict(weights["sage1"])
        self.sage2.load_state_dict(weights["sage2"])

    def forward(self, x: torch.tensor, edge_index) -> torch.Tensor:
        x = F.relu(self.sage1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.sage2(x, edge_index)
        return F.relu(x)


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
        if config.cuda:
            return inputs + rand.cuda()
        else:
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
    def mend_graph(self, org_feats, org_edges, pred_degree, gen_feats):
        new_edges = []
        if config.cuda:
            pred_degree = pred_degree.cpu()
        pred_degree = torch._cast_Int(pred_degree).detach()
        org_feats = org_feats.detach()
        fill_feats = torch.vstack((org_feats, gen_feats.view(-1, self.feat_shape)))

        for i in range(self.org_node_len):
            for j in range(min(self.num_pred, max(0, pred_degree[i]))):
                new_edges.append([i, self.org_node_len + i * self.num_pred + j])

        new_edges = torch.tensor(np.array(new_edges)).transpose(-1, 0)
        if config.cuda:
            new_edges = new_edges.cuda()
        if len(new_edges) > 0:
            fill_edges = torch.hstack((org_edges, new_edges))
        else:
            fill_edges = torch.clone(org_edges)
        return fill_edges, fill_feats

    def fill_graph(
        graph: Graph,
        predict_missing_nodes,
        predicted_features,
        num_pred=config.num_pred,
    ):
        x = graph.x.tolist()
        edges = graph.get_edges().tolist()
        node_ids = graph.node_ids.tolist()
        new_node_id = max(
            node_ids
        )  # TODO: this create problems since we have other subgraphs with the same node_id
        y = graph.y
        train_mask = graph.train_mask
        test_mask = graph.test_mask
        val_mask = graph.val_mask

        predict_missing_nodes = torch.round(predict_missing_nodes).int()
        predict_missing_nodes = np.clip(predict_missing_nodes, 0, num_pred).tolist()
        predicted_features = np.round(predicted_features.detach().numpy()).tolist()

        for i in range(graph.num_nodes):
            for j in range(predict_missing_nodes[i]):
                node_ids.append(new_node_id)
                edges[0] += [node_ids[i], new_node_id]
                edges[1] += [new_node_id, node_ids[i]]
                x.append(predicted_features[i][j])
                new_node_id += 1

        x = torch.tensor(np.array(x), dtype=torch.float32)
        edges = torch.tensor(np.array(edges))
        node_ids = torch.tensor(np.array(node_ids))

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
        return x


class LocalSage_Plus(nn.Module):
    def __init__(self, feat_shape, node_len, n_classes, node_ids):
        super(LocalSage_Plus, self).__init__()
        self.encoder_model = GNN(
            nfeat=feat_shape,
            nhid=config.hidden,
            nclass=config.latent_dim,
            dropout=config.dropout,
        )
        self.reg_model = RegModel(latent_dim=config.latent_dim)
        self.gen = Gen(
            latent_dim=config.latent_dim,
            dropout=config.dropout,
            num_pred=config.num_pred,
            feat_shape=feat_shape,
        )
        self.mend_graph = MendGraph(
            node_len=node_len,
            num_pred=config.num_pred,
            feat_shape=feat_shape,
            node_ids=node_ids,
        )
        self.classifier = GNN(
            nfeat=feat_shape,
            nhid=config.hidden,
            nclass=n_classes,
            dropout=config.dropout,
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
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats, mend_edges = self.mend_graph(feat, edges, degree, gen_feat)
        nc_pred = self.classifier(mend_feats, mend_edges)
        return degree, gen_feat, nc_pred
