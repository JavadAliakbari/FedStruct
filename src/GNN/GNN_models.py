import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, MessagePassing
from torch_geometric.utils import add_self_loops


class GNN(nn.Module):
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
                    layer_sizes[layer_num],
                    layer_sizes[layer_num + 1],
                    aggr="mean",
                )
            elif self.layer_type == "gcn":
                layer = GCNConv(
                    layer_sizes[layer_num],
                    layer_sizes[layer_num + 1],
                    # aggr="mean",
                    cached=True,
                    normalize=True,
                )
            elif self.layer_type == "gat":
                layer = GATConv(
                    layer_sizes[layer_num],
                    layer_sizes[layer_num + 1],
                    aggr="mean",
                )
            layers.append(layer)
            if layer_num < self.num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=self.dropout))

        if self.last_layer == "softmax":
            layers.append(nn.Softmax(dim=1))
        if self.last_layer == "log_softmax":
            layers.append(nn.LogSoftmax(dim=1))
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
            return nn.functional.softmax(x, dim=1)
        elif self.last_layer == "relu":
            return nn.functional.relu(x)
        else:
            return x
