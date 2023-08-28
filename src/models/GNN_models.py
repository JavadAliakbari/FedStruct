import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, MessagePassing
from torch_geometric.utils import add_self_loops

from src.utils.config_parser import Config

config = Config()


def calc_accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()


@torch.no_grad()
def calc_f1_score(pred_y, y):
    # P = pred_y[y == 1]
    # Tp = ((P == 1).sum() / len(P)).item()

    f1score = f1_score(
        pred_y.data, y.data, average="weighted", labels=np.unique(pred_y)
    )
    return f1score


@torch.no_grad()
def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    out = model(data.x, data.edge_index)
    # out = out[: len(data.test_mask)]
    label = data.y[: len(data.test_mask)]
    acc = calc_accuracy(out.argmax(dim=1)[data.test_mask], label[data.test_mask])
    return acc


class MPMLP(torch.nn.Module):
    """Graph Neural Network"""

    def __init__(
        self,
        num_gnn_layers=1,
        mlp_layer_sizes=[],
        gnn_last_layer="linear",
        mlp_last_layer="softmax",
        dropout=0.5,
        batch_normalization=False,
        multiple_features=False,
        feature_dims=0,
    ):
        super().__init__()
        self.num_gnn_layers = num_gnn_layers

        self.num_mlp_layers = len(mlp_layer_sizes) - 1
        self.mlp_layer_sizes = mlp_layer_sizes

        self.gnn_last_layer = gnn_last_layer
        self.mlp_last_layer = mlp_last_layer
        self.dropout = dropout
        self.batch_normalization = batch_normalization
        self.multiple_features = multiple_features
        self.feature_dims = feature_dims

        self.layers = self.create_models()

    def __getitem__(self, item):
        return self.layers[item]

    def create_models(self):
        self.mp = MessagePassing(aggr="mean")

        self.mlp_model = MLP(
            layer_sizes=self.mlp_layer_sizes,
            last_layer=self.mlp_last_layer,
            dropout=self.dropout,
            batch_normalization=self.batch_normalization,
        )

    def reset_parameters(self) -> None:
        self.mlp_model.reset_parameters()

    def state_dict(self):
        weights = {
            "mlp": self.mlp_model.state_dict(),
        }
        return weights

    def load_state_dict(self, weights: dict) -> None:
        self.mlp_model.load_state_dict(weights["mlp"])

    def gnn_step(self, h, edge_index) -> None:
        a = 0.1
        x = h
        edge_index_ = add_self_loops(edge_index)[0]
        for _ in range(self.num_gnn_layers):
            x = self.mp.propagate(edge_index_, x=x)
            x = (1 - a) * x + a * h

        if self.gnn_last_layer == "softmax":
            return F.softmax(x, dim=1)
        elif self.gnn_last_layer == "relu":
            return F.relu(x)
        else:
            return x

    def mlp_step(self, h) -> None:
        return self.mlp_model(h)

    def forward(self, x, edge_index):
        h = self.mlp_step(x)
        out = self.gnn_step(h, edge_index)
        # h = self.gnn_step(x, edge_index)
        # out = self.mlp_step(h)
        return out


class GNNMLP(torch.nn.Module):
    """Graph Neural Network"""

    def __init__(
        self,
        gnn_layer_sizes,
        mlp_layer_sizes=[],
        gnn_last_layer="linear",
        mlp_last_layer="softmax",
        dropout=0.5,
        batch_normalization=False,
        multiple_features=False,
        feature_dims=0,
    ):
        super().__init__()
        self.num_gnn_layers = len(gnn_layer_sizes) - 1
        self.gnn_layer_sizes = gnn_layer_sizes

        self.num_mlp_layers = len(mlp_layer_sizes) - 1
        self.mlp_layer_sizes = mlp_layer_sizes

        self.gnn_last_layer = gnn_last_layer
        self.mlp_last_layer = mlp_last_layer
        self.dropout = dropout
        self.batch_normalization = batch_normalization
        self.multiple_features = multiple_features
        self.feature_dims = feature_dims

        self.layers = self.create_models()

    def __getitem__(self, item):
        return self.layers[item]

    def create_models(self):
        self.gnn_model = GNN(
            layer_sizes=self.gnn_layer_sizes,
            last_layer=self.gnn_last_layer,
            layer_type=config.model.gnn_layer_type,
            dropout=self.dropout,
            batch_normalization=self.batch_normalization,
            multiple_features=self.multiple_features,
        )

        self.mlp_model = MLP(
            layer_sizes=self.mlp_layer_sizes,
            last_layer=self.mlp_last_layer,
            dropout=self.dropout,
            batch_normalization=False,
        )

    def reset_parameters(self) -> None:
        self.gnn_model.reset_parameters()
        self.mlp_model.reset_parameters()

    def state_dict(self):
        weights = {
            "gnn": self.gnn_model.state_dict(),
            "mlp": self.mlp_model.state_dict(),
        }
        return weights

    def load_state_dict(self, weights: dict) -> None:
        self.gnn_model.load_state_dict(weights["gnn"])
        self.mlp_model.load_state_dict(weights["mlp"])

    def gnn_step(self, h, edge_index) -> None:
        return self.gnn_model(h, edge_index)

    def mlp_step(self, h) -> None:
        return self.mlp_model(h)

    def forward(self, x, edge_index):
        h = self.gnn_step(x, edge_index)
        out = self.mlp_step(h)
        return out


class GNN(torch.nn.Module):
    """Graph Neural Network"""

    def __init__(
        self,
        layer_sizes,
        last_layer="linear",
        layer_type="sage",
        dropout=0.5,
        batch_normalization=False,
        multiple_features=False,
        feature_dims=0,
    ):
        super().__init__()
        self.num_layers = len(layer_sizes) - 1
        # self.layer_sizes = layer_sizes

        self.last_layer = last_layer
        self.layer_type = layer_type
        self.dropout = dropout
        self.batch_normalization = batch_normalization
        self.multiple_features = multiple_features
        self.feature_dims = feature_dims

        self.layers = self.create_models(layer_sizes)

    def __getitem__(self, item):
        return self.layers[item]

    def create_models(self, layer_sizes):
        if self.multiple_features:
            self.mp_layer = nn.Linear(self.feature_dims, 1, bias=False)

        if self.batch_normalization:
            self.batch_layer = nn.BatchNorm1d(layer_sizes[0])

        gnn_layers = nn.ParameterList()
        for layer_num in range(self.num_layers):
            if self.layer_type == "sage":
                layer = SAGEConv(
                    layer_sizes[layer_num], layer_sizes[layer_num + 1], aggr="mean"
                )
            elif self.layer_type == "gcn":
                layer = GCNConv(
                    layer_sizes[layer_num], layer_sizes[layer_num + 1], aggr="mean"
                )
            elif self.layer_type == "gat":
                layer = GATConv(
                    layer_sizes[layer_num], layer_sizes[layer_num + 1], aggr="mean"
                )
            gnn_layers.append(layer)

        return gnn_layers

    def reset_parameters(self) -> None:
        if self.batch_normalization:
            self.batch_layer.reset_parameters()
        if self.multiple_features:
            self.mp_layer.reset_parameters()
        for layers in self.layers:
            layers.reset_parameters()

    def state_dict(self):
        weights = {}
        if self.batch_normalization:
            weights["batch_layer"] = self.batch_layer.state_dict()
        if self.multiple_features:
            weights["mp_layer"] = self.mp_layer.state_dict()

        for id, layer in enumerate(self.layers):
            weights[f"layer{id}"] = layer.state_dict()
        return weights

    def load_state_dict(self, weights: dict) -> None:
        if self.batch_normalization:
            self.batch_layer.load_state_dict(weights["batch_layer"])
        if self.multiple_features:
            self.mp_layer.load_state_dict(weights["mp_layer"])

        for id, layer in enumerate(self.layers):
            layer.load_state_dict(weights[f"layer{id}"])

    def normalize_mp_weights(self):
        with torch.no_grad():
            w = self.mp_layer.weight.data
            w = (w - torch.min(w)) / (torch.max(w) - torch.min(w))
            self.mp_layer.weight.data = w / w.sum()

    def step(self, h, edge_index, layer_id):
        layer = self.layers[layer_id]
        h = layer(h, edge_index).relu()
        h = F.dropout(h, p=self.dropout, training=self.training)

        return h

    def forward(self, x, edge_index):
        h = x
        if self.multiple_features:
            self.normalize_mp_weights()
            h = self.mp_layer(h).squeeze()

        if self.batch_normalization:
            h = self.batch_layer(h)

        for layer in self.layers[:-1]:
            h = layer(h, edge_index).relu()
            h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.layers[-1](h, edge_index)

        if self.last_layer == "softmax":
            return F.softmax(out, dim=1)
        elif self.last_layer == "relu":
            return F.relu(out)
        else:
            return out


class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes,
        last_layer="softmax",
        dropout=0.5,
        batch_normalization=False,
    ):
        super().__init__()

        self.num_layers = len(layer_sizes) - 1
        self.dropout = dropout
        self.last_layer = last_layer
        self.batch_normalization = batch_normalization

        self.layers = self.create_models(layer_sizes)

        self.default_weights = self.state_dict()

    def __getitem__(self, item):
        return self.layers[item]

    def create_models(self, layer_sizes):
        if self.batch_normalization:
            self.batch_layer = nn.BatchNorm1d(layer_sizes[0])

        layers = nn.ParameterList()
        for layer_num in range(self.num_layers):
            layer = nn.Linear(layer_sizes[layer_num], layer_sizes[layer_num + 1])
            layers.append(layer)

        return layers

    def reset_parameters(self) -> None:
        self.load_state_dict(self.default_weights)
        # if self.batch_normalization:
        #     self.batch_layer.reset_parameters()
        # for layers in self.layers:
        #     layers.reset_parameters()

    def state_dict(self):
        weights = {}
        if self.batch_normalization:
            weights["batch_layer"] = self.batch_layer.state_dict()
        for id, layer in enumerate(self.layers):
            weights[f"layer{id}"] = layer.state_dict()
        return weights

    def load_state_dict(self, weights: dict) -> None:
        if self.batch_normalization:
            self.batch_layer.load_state_dict(weights["batch_layer"])
        for id, layer in enumerate(self.layers):
            layer.load_state_dict(weights[f"layer{id}"])

    def step(self, h, layer_id):
        layer = self.layers[layer_id]
        h = layer(h).relu()
        h = F.dropout(h, p=self.dropout, training=self.training)

        return h

    def forward(self, x):
        h = x
        if self.batch_normalization:
            h = self.batch_layer(h)
        for layer in self.layers[:-1]:
            h = layer(h).relu()
            h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.layers[-1](h)

        if self.last_layer == "softmax":
            return F.softmax(out, dim=1)
        elif self.last_layer == "relu":
            return F.relu(out)
        else:
            return out

    def fit(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        epochs=1,
        verbose=False,
        plot=False,
    ):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.model.lr,
            weight_decay=config.model.weight_decay,
        )

        self.train()
        train_acc_list = []
        val_acc_list = []
        for epoch in range(epochs + 1):
            # Training
            optimizer.zero_grad()
            y_pred = self(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            # Print metrics every 10 epochs
            if verbose and x_val is not None:
                acc = calc_accuracy(y_pred.argmax(dim=1), y_train)
                f1_score = calc_f1_score(y_pred.argmax(dim=1), y_train)
                # Validation
                y_pred_val = self(x_val)
                val_loss = criterion(y_pred_val, y_val)
                val_acc = calc_accuracy(y_pred_val.argmax(dim=1), y_val)
                val_f1_score = calc_f1_score(y_pred_val.argmax(dim=1), y_val)

                if epoch % 10 == 0:
                    print(
                        f"Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc:"
                        f" {acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | "
                        f"F1 Score: {f1_score*100:.2f}% | "
                        f"Val Acc: {val_acc*100:.2f}% | "
                        f"Val F1 Score: {val_f1_score*100:.2f}%",
                        end="\r",
                    )

                train_acc_list.append(acc)
                val_acc_list.append(val_acc)

        if verbose:
            print("\n")

        if plot:
            plt.figure()
            plt.plot(train_acc_list)
            plt.plot(val_acc_list)

        self.eval()
        y_pred_val = self(x_val)
        val_loss = criterion(y_pred_val, y_val)
        val_acc = calc_accuracy(y_pred_val.argmax(dim=1), y_val)

        return val_acc, val_loss
        # return loss, val_loss, acc, val_acc, TP, val_TP

    def test(self, x, y):
        self.eval()
        out = self(x)
        test_accuracy = calc_accuracy(out.argmax(dim=1), y)
        return test_accuracy
