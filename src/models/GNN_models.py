import os

from sklearn.utils import compute_class_weight

from src.utils.utils import *

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, MessagePassing
from torch_geometric.utils import add_self_loops

from src.utils.config_parser import Config

path = os.environ.get("CONFIG_PATH")
config = Config(path)


class ModelSpecs:
    def __init__(
        self,
        type="GNN",
        layer_sizes=[],
        final_activation_function="linear",  # can be None, "layer", "batch", "instance"
        # dropout=0.5,
        normalization=None,
        # gnn_layer_type="sage",
        num_layers=None,
    ):
        self.type = type
        self.layer_sizes = layer_sizes
        self.final_activation_function = final_activation_function
        # self.dropout = dropout
        self.normalization = normalization
        # self.gnn_layer_type = gnn_layer_type
        if num_layers is None:
            self.num_layers = len(self.layer_sizes) - 1
        else:
            self.num_layers = num_layers


class ModelBinder(torch.nn.Module):
    def __init__(
        self,
        models_specs=[],
    ):
        super().__init__()
        self.models_specs = models_specs

        self.models = self.create_models()

    def __getitem__(self, item):
        return self.models[item]

    def create_models(self):
        models = nn.ParameterList()
        model_propertises: ModelSpecs
        for model_propertises in self.models_specs:
            if model_propertises.type == "GNN":
                model = GNN(
                    layer_sizes=model_propertises.layer_sizes,
                    last_layer=model_propertises.final_activation_function,
                    layer_type=config.model.gnn_layer_type,
                    dropout=config.model.dropout,
                    normalization=model_propertises.normalization,
                )
            elif model_propertises.type == "MLP":
                model = MLP(
                    layer_sizes=model_propertises.layer_sizes,
                    last_layer=model_propertises.final_activation_function,
                    dropout=config.model.dropout,
                    normalization=model_propertises.normalization,
                )
            elif model_propertises.type == "DGCN":
                model = DGCN(
                    num_layers=model_propertises.num_layers,
                    last_layer=model_propertises.final_activation_function,
                    aggr="mean",
                    a=0.0,
                )

            models.append(model)

        return models

    def reset_parameters(self) -> None:
        for model in self.models:
            model.reset_parameters()

    def state_dict(self):
        weights = {}
        for id, model in enumerate(self.models):
            weights[f"model{id}"] = model.state_dict()
        return weights

    def load_state_dict(self, weights: dict) -> None:
        for id, model in enumerate(self.models):
            model.load_state_dict(weights[f"model{id}"])

    def get_grads(self):
        model_parameters = list(self.parameters())
        grads = [parameter.grad for parameter in model_parameters]

        return grads

    def set_grads(self, grads):
        model_parameters = list(self.parameters())
        for grad, parameter in zip(grads, model_parameters):
            parameter.grad = grad

    def step(self, model: GCNConv, h, edge_index=None, edge_weight=None) -> None:
        if model.type_ == "MLP":
            return model(h)
        else:
            return model(h, edge_index, edge_weight)

    def forward(self, x, edge_index=None, edge_weight=None):
        h = x
        for model in self.models:
            h = self.step(model, h, edge_index, edge_weight)
        return h


class DGCN(MessagePassing):
    def __init__(
        self,
        aggr="mean",
        num_layers=1,
        a=0,
        last_layer="linear",
        normalization=None,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)
        self.type_ = DGCN
        self.num_layers = num_layers
        self.last_layer = last_layer
        self.a = a
        # self.normalization = normalization

        # self.layers = nn.ParameterList()
        # if self.normalization:
        #     batch_layer = nn.BatchNorm1d(layer_sizes[0], affine=True)
        #     # batch_layer = nn.LayerNorm(layer_sizes[0])
        #     self.layers.append(batch_layer)

    def reset_parameters(self) -> None:
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, weights: dict) -> None:
        pass

    def get_grads(self):
        return []

    def forward(self, h, edge_index) -> None:
        x = h
        edge_index_ = add_self_loops(edge_index)[0]
        for _ in range(self.num_layers):
            x = self.propagate(edge_index_, x=x)
            x = (1 - self.a) * x + self.a * h

        if self.last_layer == "softmax":
            return F.softmax(x, dim=1)
        elif self.last_layer == "relu":
            return F.relu(x)
        else:
            return x


class GNN(torch.nn.Module):
    """Graph Neural Network"""

    def __init__(
        self,
        layer_sizes,
        last_layer="linear",
        layer_type="sage",
        dropout=0.5,
        normalization=None,
        multiple_features=False,
        feature_dims=0,
    ):
        super().__init__()
        self.type_ = "GNN"
        self.num_layers = len(layer_sizes) - 1
        # self.layer_sizes = layer_sizes

        self.last_layer = last_layer
        self.layer_type = layer_type
        self.dropout = dropout
        self.normalization = normalization
        self.multiple_features = multiple_features
        self.feature_dims = feature_dims

        self.layers = self.create_models(layer_sizes)
        # self.net = nn.Sequential(*self.layers)

    def __getitem__(self, item):
        return self.layers[item]

    def create_models(self, layer_sizes):
        layers = nn.ParameterList()
        if self.multiple_features:
            mp_layer = nn.Linear(self.feature_dims, 1, bias=False)
            layers.append(mp_layer)
            layers.append(nn.Flatten(start_dim=1))

        if self.normalization == "batch":
            norm_layer = nn.BatchNorm1d(
                layer_sizes[0], affine=True, track_running_stats=False
            )
            layers.append(norm_layer)
        elif self.normalization == "layer":
            norm_layer = nn.LayerNorm(layer_sizes[0])
            layers.append(norm_layer)
        elif self.normalization == "instance":
            norm_layer = nn.InstanceNorm1d(layer_sizes[0])
            layers.append(norm_layer)

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
            layers.append(layer)
            if layer_num < self.num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=self.dropout))

        if self.last_layer == "softmax":
            layers.append(nn.Softmax(dim=1))
        elif self.last_layer == "relu":
            layers.append(nn.ReLU())

        return layers

    def reset_parameters(self) -> None:
        for layer in self.layers:
            try:
                layer.reset_parameters()
            except:
                pass

    def state_dict(self):
        weights = {}
        for id, layer in enumerate(self.layers):
            if (
                isinstance(layer, nn.LayerNorm)
                or isinstance(layer, nn.BatchNorm1d)
                or isinstance(layer, nn.InstanceNorm1d)
            ):
                continue
            weights[f"layer{id}"] = layer.state_dict()
        return weights

    def load_state_dict(self, weights: dict) -> None:
        for id, layer in enumerate(self.layers):
            if (
                isinstance(layer, nn.LayerNorm)
                or isinstance(layer, nn.BatchNorm1d)
                or isinstance(layer, nn.InstanceNorm1d)
            ):
                continue
            layer.load_state_dict(weights[f"layer{id}"])

    def get_grads(self):
        model_parameters = list(self.parameters())
        grads = [parameter.grad for parameter in model_parameters]

        return grads

    def set_grads(self, grads):
        model_parameters = list(self.parameters())
        for grad, parameter in zip(grads, model_parameters):
            parameter.grad = grad

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for layer in self.layers:
            if isinstance(layer, MessagePassing):
                h = layer(h, edge_index, edge_weight)
            else:
                h = layer(h)

        return h


class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes,
        last_layer="softmax",
        dropout=0.5,
        normalization=None,
    ):
        super().__init__()
        self.type_ = "MLP"
        self.num_layers = len(layer_sizes) - 1
        self.dropout = dropout
        self.last_layer = last_layer
        self.normalization = normalization

        self.layers = self.create_models(layer_sizes)
        # self.net = nn.Sequential(*self.layers)

        # self.default_weights = self.state_dict()
        # self.default_weights = deepcopy(self.state_dict())

    def __getitem__(self, item):
        return self.layers[item]

    def create_models(self, layer_sizes):
        layers = nn.ParameterList()

        if self.normalization == "batch":
            norm_layer = nn.BatchNorm1d(
                layer_sizes[0], affine=True, track_running_stats=False
            )
            layers.append(norm_layer)
        elif self.normalization == "layer":
            norm_layer = nn.LayerNorm(layer_sizes[0])
            layers.append(norm_layer)
        elif self.normalization == "instance":
            norm_layer = nn.InstanceNorm1d(layer_sizes[0])
            layers.append(norm_layer)

        for layer_num in range(self.num_layers):
            layer = nn.Linear(layer_sizes[layer_num], layer_sizes[layer_num + 1])
            layers.append(layer)
            if layer_num < self.num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=self.dropout))

        if self.last_layer == "softmax":
            layers.append(nn.Softmax(dim=1))
        elif self.last_layer == "relu":
            layers.append(nn.ReLU())
        elif self.last_layer == "tanh":
            layers.append(nn.Tanh())
        elif self.last_layer == "sigmoid":
            layers.append(nn.Sigmoid())

        return layers

    def reset_parameters(self) -> None:
        # self.load_state_dict(self.default_weights)
        for layer in self.layers:
            try:
                layer.reset_parameters()
            except:
                pass

    def state_dict(self):
        weights = {}
        for id, layer in enumerate(self.layers):
            if (
                isinstance(layer, nn.LayerNorm)
                or isinstance(layer, nn.BatchNorm1d)
                or isinstance(layer, nn.InstanceNorm1d)
            ):
                continue
            weights[f"layer{id}"] = layer.state_dict()
        return weights

    def load_state_dict(self, weights: dict) -> None:
        for id, layer in enumerate(self.layers):
            if (
                isinstance(layer, nn.LayerNorm)
                or isinstance(layer, nn.BatchNorm1d)
                or isinstance(layer, nn.InstanceNorm1d)
            ):
                continue
            layer.load_state_dict(weights[f"layer{id}"])

    def get_grads(self):
        model_parameters = list(self.parameters())
        grads = [parameter.grad for parameter in model_parameters]

        return grads

    def set_grads(self, grads):
        model_parameters = list(self.parameters())
        for grad, parameter in zip(grads, model_parameters):
            parameter.grad = grad

    def train(self, mode: bool = True):
        super().train(mode)
        for layer in self.layers:
            layer.train(mode)

    def val(self):
        super().val()
        for layer in self.layers:
            layer.val()

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)

        return h
        # return self.net(x)

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

    @torch.no_grad()
    def test(self, x, y):
        self.eval()
        out = self(x)
        test_accuracy = calc_accuracy(out.argmax(dim=1), y)
        return test_accuracy
