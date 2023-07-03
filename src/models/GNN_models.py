import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.utils import config
from src.utils.graph import Graph


def calc_accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()


@torch.no_grad()
def test(model, data):
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
        self.sage1 = SAGEConv(dim_in, dim_h[0])
        self.sage2 = SAGEConv(dim_h[0], dim_h[1])
        self.sage3 = SAGEConv(dim_h[1], dim_out)
        # self.dense = Linear(dim_h[1], dim_out)

        self.dropout = dropout
        self.last_layer = last_layer

    def reset_parameters(self) -> None:
        self.sage1.reset_parameters()
        self.sage2.reset_parameters()
        self.sage3.reset_parameters()

    def state_dict(self):
        weights = {}
        # for leyer_name, layer_weights in self.sage1.named_parameters():
        weights["sage1"] = self.sage1.state_dict()
        weights["sage2"] = self.sage2.state_dict()
        weights["sage3"] = self.sage3.state_dict()
        return weights

    def load_state_dict(self, weights: dict) -> None:
        self.sage1.load_state_dict(weights["sage1"])
        self.sage2.load_state_dict(weights["sage2"])
        self.sage3.load_state_dict(weights["sage3"])

    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index).relu()
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.sage2(h, edge_index).relu()
        out = self.sage3(h, edge_index)
        
        if self.last_layer == "softmax":
            return F.softmax(out, dim=1)
        elif self.last_layer == "relu":
            return F.relu(out)
        else:
            return out