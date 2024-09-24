import torch

from src import *
from src.classifier import Classifier
from src.models.model_binders import ModelBinder, ModelSpecs
from src.utils.graph import Graph
from src.utils.data import Data
from src.FedGCN.gnn_models import GCN, GCN_arxiv, SAGE_products


class FedGCNClassifier(Classifier):
    def __init__(self, graph: Graph):
        self.graph: Graph | Data | None = graph
        self.create_model()
        self.create_optimizer()

    def create_model(self):
        # if config.fedgcn.gnn_type == "gcn":
        #     self.model = GCN(
        #         nfeat=self.graph.x.shape[1],
        #         nhid=config.fedgcn.args_hidden,
        #         nclass=self.graph.num_classes,
        #         dropout=0.5,
        #         NumLayers=config.fedgcn.num_layers,
        #     ).to(device)
        # elif config.fedgcn.gnn_type == "sage":
        gnn_layer_sizes = [self.graph.num_features] + config.fedgcn.gnn_layer_sizes
        if len(config.fedgcn.mlp_layer_sizes) > 0:
            mlp_layer_sizes = (
                [gnn_layer_sizes[-1]]
                + config.fedgcn.gnn_layer_sizes
                + [self.graph.num_classes]
            )
        else:
            gnn_layer_sizes.append(self.graph.num_classes)

        if config.fedgcn.gnn_type == "gcn":
            normalization = None
        else:
            normalization = "layer"
        model_specs = [
            ModelSpecs(
                type="GNN",
                layer_sizes=gnn_layer_sizes,
                final_activation_function="linear",
                dropout=0.5,
                # final_activation_function="relu",
                gnn_layer_type=config.fedgcn.gnn_type,
                normalization=normalization,
                # normalization="batch",
            ),
        ]
        if len(config.fedgcn.mlp_layer_sizes) > 0:
            model_specs.append(
                ModelSpecs(
                    type="MLP",
                    layer_sizes=mlp_layer_sizes,
                    final_activation_function="linear",
                    normalization=None,
                ),
            )
        self.model: ModelBinder = ModelBinder(model_specs)
        self.model.to(device)

    def create_optimizer(self):
        parameters = self.parameters()
        if len(parameters) == 0:
            return

        if config.fedgcn.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                parameters,
                lr=config.fedgcn.lr,
                weight_decay=config.fedgcn.weight_decay,
            )
        elif config.fedgcn.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                parameters,
                lr=config.fedgcn.lr,
                weight_decay=config.fedgcn.weight_decay,
            )

    def get_embeddings(self):
        return self.model(self.graph.x, self.graph.edge_index)

    def __call__(self):
        return self.get_embeddings()

    def get_prediction(self):
        H = self.get_embeddings()
        if config.fedgcn.final_activation == "softmax":
            y_pred = torch.nn.functional.softmax(H, dim=1)
        elif config.fedgcn.final_activation == "log_softmax":
            y_pred = torch.nn.functional.log_softmax(H, dim=1)
        return y_pred

    def train_step(self, eval_=True):
        if config.fedgcn.final_activation == "log_softmax":
            loss_function = "log_likelihood"
        elif config.fedgcn.final_activation == "softmax":
            loss_function = "cross_entropy"
        train_loss, train_acc = Classifier.calc_mask_metric(
            self, mask="train", loss_function=loss_function
        )
        train_loss.backward()

        if eval_:
            (test_acc,) = Classifier.calc_mask_metric(self, mask="test", metric="acc")
            if self.graph.val_mask is not None:
                val_loss, val_acc = Classifier.calc_mask_metric(
                    self, mask="val", loss_function=loss_function
                )
                return train_loss.item(), train_acc, val_loss.item(), val_acc, test_acc
            else:
                return train_loss.item(), train_acc, 0, 0, test_acc

        else:
            return train_loss.item(), train_acc
