import torch

from src import *
from src.classifier import Classifier
from src.utils.graph import Graph
from src.utils.data import Data
from src.FedGCN.gnn_models import GCN, GCN_arxiv, SAGE_products


class FedGCNClassifier(Classifier):
    def __init__(self, graph: Graph):
        self.graph: Graph | Data | None = graph
        self.create_model()
        self.create_optimizer()

    def create_model(self):
        if config.dataset.dataset_name == "ogbn-arxiv":
            self.model = GCN_arxiv(
                nfeat=self.graph.x.shape[1],
                nhid=config.fedgcn.args_hidden,
                nclass=self.graph.num_classes,
                dropout=0.5,
                NumLayers=config.fedgcn.num_layers,
            ).to(device)
        elif config.dataset.dataset_name == "ogbn-products":
            self.model = SAGE_products(
                nfeat=self.graph.x.shape[1],
                nhid=config.fedgcn.args_hidden,
                nclass=self.graph.num_classes,
                dropout=0.5,
                NumLayers=config.fedgcn.num_layers,
            ).to(device)
        else:
            self.model = GCN(
                nfeat=self.graph.x.shape[1],
                nhid=config.fedgcn.args_hidden,
                nclass=self.graph.num_classes,
                dropout=0.5,
                NumLayers=config.fedgcn.num_layers,
            ).to(device)

    def create_optimizer(self):
        parameters = self.parameters()
        if len(parameters) == 0:
            return

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.fedgcn.lr,
            weight_decay=config.fedgcn.weight_decay,
        )

    def get_embeddings(self):
        return self.model(self.graph.x, self.graph.edge_index)

    def __call__(self):
        return self.get_embeddings()

    def get_prediction(self):
        return self.get_embeddings()

    def train_step(self, eval_=True):
        train_loss, train_acc = Classifier.calc_mask_metric(
            self, mask="train", loss_function="log_likelihood"
        )
        train_loss.backward()

        if eval_:
            (test_acc,) = Classifier.calc_mask_metric(self, mask="test", metric="acc")
            if self.graph.val_mask is not None:
                val_loss, val_acc = Classifier.calc_mask_metric(
                    self, mask="val", loss_function="log_likelihood"
                )
                return train_loss.item(), train_acc, val_loss.item(), val_acc, test_acc
            else:
                return train_loss.item(), train_acc, 0, 0, test_acc

        else:
            return train_loss.item(), train_acc
