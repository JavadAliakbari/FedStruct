import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from sklearn.metrics import f1_score


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


class GNN(torch.nn.Module):
    """Graph Neural Network"""

    def __init__(
        self,
        in_dims,
        out_dims=None,
        dropout=0.5,
        linear_layer=False,
        last_layer="softmax",
        batch_normalization=False,
    ):
        super().__init__()
        if out_dims is not None:
            assert len(in_dims) == len(out_dims), "Number of layers must be equal"
            input_dims = in_dims
            output_dims = out_dims
        else:
            input_dims = in_dims[:-1]
            output_dims = in_dims[1:]

        self.num_layers = len(input_dims)
        self.linear_layer = linear_layer
        self.dropout = dropout
        self.last_layer = last_layer
        self.batch_normalization = batch_normalization

        self.layers = self.create_models(input_dims, output_dims)

    def __getitem__(self, item):
        return self.layers[item]

    def create_models(self, in_dims, out_dims):
        layers = nn.ParameterList()
        if self.batch_normalization:
            self.batch_layer = nn.BatchNorm1d(in_dims[0])
        for layer_num in range(self.num_layers):
            if layer_num == self.num_layers - 1 and self.linear_layer:
                layer = nn.Linear(in_dims[layer_num], out_dims[layer_num])
            else:
                layer = SAGEConv(in_dims[layer_num], out_dims[layer_num], aggr="mean")
                # layer = GCNConv(in_dims[layer_num], out_dims[layer_num], aggr="mean")
                # layer = GATConv(in_dims[layer_num], out_dims[layer_num], aggr="mean")
            layers.append(layer)

        return layers

    def reset_parameters(self) -> None:
        if self.batch_normalization:
            self.batch_layer.reset_parameters()
        for layers in self.layers:
            layers.reset_parameters()

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

    def forward(self, x, edge_index):
        h = x
        if self.batch_normalization:
            h = self.batch_layer(h)

        for layer in self.layers[:-1]:
            h = layer(h, edge_index).relu()
            h = F.dropout(h, p=self.dropout, training=self.training)

        if self.linear_layer:
            out = self.layers[-1](h)
        else:
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
        dropout=0.5,
        softmax=True,
        batch_normalization=False,
    ):
        super().__init__()

        self.num_layers = len(layer_sizes) - 1
        self.dropout = dropout
        self.softmax = softmax
        self.batch_normalization = batch_normalization

        self.layers = self.create_models(layer_sizes)

    def __getitem__(self, item):
        return self.layers[item]

    def create_models(self, layer_sizes):
        layers = nn.ParameterList()
        if self.batch_normalization:
            self.batch_layer = nn.BatchNorm1d(layer_sizes[0])
        for layer_num in range(self.num_layers):
            layer = nn.Linear(layer_sizes[layer_num], layer_sizes[layer_num + 1])
            layers.append(layer)

        return layers

    def reset_parameters(self) -> None:
        if self.batch_normalization:
            self.batch_layer.reset_parameters()
        for layers in self.layers:
            layers.reset_parameters()

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

    def forward(self, x):
        h = x
        if self.batch_normalization:
            h = self.batch_layer(h)
        for layer in self.layers[:-1]:
            h = layer(h).relu()
            h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.layers[-1](h)

        if self.softmax:
            out = F.softmax(out, dim=1)
        return out

    def fit(self, x, y, masks, epochs, verbose=False):
        train_mask, val_mask, test_mask = masks
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)

        self.train()
        for epoch in range(epochs + 1):
            # Training
            optimizer.zero_grad()
            out = self(x)
            loss = criterion(out[train_mask], y[train_mask])
            acc = calc_accuracy(out[train_mask].argmax(dim=1), y[train_mask])
            f1_score = calc_f1_score(out[train_mask].argmax(dim=1), y[train_mask])
            loss.backward()
            optimizer.step()

            # Validation
            val_loss = criterion(out[val_mask], y[val_mask])
            val_acc = calc_accuracy(out[val_mask].argmax(dim=1), y[val_mask])
            val_f1_score = calc_f1_score(out[val_mask].argmax(dim=1), y[val_mask])

            # Print metrics every 10 epochs
            if verbose:
                if epoch % 10 == 0:
                    print(
                        f"Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc:"
                        f" {acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | "
                        f"F1 Score: {f1_score*100:.2f}% | "
                        f"Val Acc: {val_acc*100:.2f}% | "
                        f"Val F1 Score: {val_f1_score*100:.2f}%",
                        end="\r",
                    )

        if verbose:
            print("\n")

        return val_acc, val_loss
        # return loss, val_loss, acc, val_acc, TP, val_TP

    def test(self, x, y):
        out = self(x)
        test_accuracy = calc_accuracy(out.argmax(dim=1), y)
        return test_accuracy
