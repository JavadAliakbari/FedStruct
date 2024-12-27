import torch.nn as nn
import torch.nn.functional as F

from src.FedPub.utils import *


class GCN(nn.Module):
    def __init__(self, n_feat=10, n_dims=128, n_clss=10):
        super().__init__()
        self.n_feat = n_feat
        self.n_dims = n_dims
        self.n_clss = n_clss

        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(self.n_feat, self.n_dims, cached=False)
        self.conv2 = GCNConv(self.n_dims, self.n_dims, cached=False)
        self.clsif = nn.Linear(self.n_dims, self.n_clss)

    def forward(self, data, is_proxy=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if is_proxy == True:
            return x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.clsif(x)
        return x


class MaskedGCN(nn.Module):
    def __init__(self, n_feat=10, n_dims=128, n_clss=10, l1=1e-3):
        super().__init__()
        self.n_feat = n_feat
        self.n_dims = n_dims
        self.n_clss = n_clss

        from src.FedPub.layers import MaskedGCNConv, MaskedLinear

        self.conv1 = MaskedGCNConv(self.n_feat, self.n_dims, cached=False, l1=l1)
        self.conv2 = MaskedGCNConv(self.n_dims, self.n_dims, cached=False, l1=l1)
        self.clsif = MaskedLinear(self.n_dims, self.n_clss, l1=l1)

    def forward(self, data, is_proxy=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if is_proxy == True:
            return x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.clsif(x)
        # x = F.softmax(x, dim=-1)
        return x
