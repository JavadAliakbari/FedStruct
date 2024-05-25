import os

import torch
from torch_geometric.loader import NeighborLoader

from src.GNN.DGCN import DGCN, SDGCN
from src.GNN.fGNN import FGNN
from src.GNN.sGNN import SGNNMaster, SGNNSlave
from src.utils.utils import *
from src.utils.graph import AGraph, Data, Graph
from src.classifier import Classifier
from src.utils.config_parser import Config

dev = os.environ.get("device", "cpu")
device = torch.device(dev)

path = os.environ.get("CONFIG_PATH")
config = Config(path)


class FedClassifier(Classifier):
    def __init__(self, graph: Data):
        super().__init__()
        self.graph = graph
        self.f_model: Classifier = None
        self.s_model: Classifier = None

    def state_dict(self):
        weights = super().state_dict()
        weights["fmodel"] = self.f_model.state_dict()
        weights["smodel"] = self.s_model.state_dict()
        return weights

    def load_state_dict(self, weights):
        super().load_state_dict(weights)
        self.f_model.load_state_dict(weights["fmodel"])
        self.s_model.load_state_dict(weights["smodel"])

    def get_grads(self, just_SFV=False):
        grads = super().get_grads(just_SFV)
        grads["fmodel"] = self.f_model.get_grads(just_SFV)
        grads["smodel"] = self.s_model.get_grads(just_SFV)
        return grads

    def set_grads(self, grads):
        super().set_grads(grads)
        self.f_model.set_grads(grads["fmodel"])
        self.s_model.set_grads(grads["smodel"])

    def reset_parameters(self):
        super().reset_parameters()
        self.f_model.reset_parameters()
        self.s_model.reset_parameters()

    def parameters(self):
        parameters = super().parameters()
        parameters += self.f_model.parameters()
        parameters += self.s_model.parameters()
        return parameters

    def train(self, mode: bool = True):
        self.f_model.train(mode)
        self.s_model.train(mode)

    def eval(self):
        self.f_model.eval()
        self.s_model.eval()

    def zero_grad(self, set_to_none=False):
        self.f_model.zero_grad(set_to_none=set_to_none)
        self.s_model.zero_grad(set_to_none=set_to_none)

    def restart(self):
        super().restart()
        self.f_model = None
        self.s_model = None

    def reset(self):
        super().reset()
        self.f_model.reset()
        self.s_model.reset()

    def create_model(self):
        raise NotImplementedError

    def get_embeddings(self):
        H = self.f_model()
        S = self.s_model()
        O = H + S
        return O

    def __call__(self):
        return self.get_embeddings()

    def get_prediction(self):
        O = self.get_embeddings()
        y_pred = torch.nn.functional.softmax(O, dim=1)
        return y_pred

    def train_step(self, eval_=True):
        y_pred = self.get_prediction()
        y = self.graph.y
        train_mask, val_mask, _ = self.graph.get_masks()

        train_loss, train_acc = calc_metrics(y, y_pred, train_mask)
        train_loss.backward(retain_graph=True)

        if eval_:
            self.eval()
            y_pred_val = self.get_prediction()
            val_loss, val_acc = calc_metrics(y, y_pred_val, val_mask)

            y_pred_f = self.f_model()
            y_pred_s = self.s_model()
            _, val_acc_f = calc_metrics(y, y_pred_f, val_mask)
            _, val_acc_s = calc_metrics(y, y_pred_s, val_mask)

            return (
                train_loss.item(),
                train_acc,
                val_loss.item(),
                val_acc,
                val_acc_f,
                val_acc_s,
            )
        else:
            return train_loss.item(), train_acc

    @torch.no_grad()
    def calc_test_accuracy(self, metric="acc"):
        self.eval()

        y = self.graph.y
        test_mask = self.graph.test_mask

        y_pred = self.get_prediction()
        test_loss, test_acc = calc_metrics(y, y_pred, test_mask)

        if metric == "acc":
            return (test_acc,)
        # elif metric == "f1":
        #     return test_f1_score
        else:
            return (test_loss.item(),)

    def calc_test_accuracy_f(self, metric="acc"):
        self.eval()

        y = self.graph.y
        test_mask = self.graph.test_mask

        y_pred_f = self.f_model.get_prediction()
        test_loss_f, test_acc_f = calc_metrics(y, y_pred_f, test_mask)

        if metric == "acc":
            return test_acc_f
        # elif metric == "f1":
        #     return test_f1_score
        else:
            return test_loss_f

    def calc_test_accuracy_s(self, metric="acc"):
        self.eval()

        y = self.graph.y
        test_mask = self.graph.test_mask

        y_pred_s = self.s_model.get_prediction()
        test_loss_s, test_acc_s = calc_metrics(y, y_pred_s, test_mask)

        if metric == "acc":
            return test_acc_s
        # elif metric == "f1":
        #     return test_f1_score
        else:
            return test_loss_s.item()


class FedGNNSlave(FedClassifier):
    def __init__(self, graph: Graph, server_embedding_func):
        super().__init__(graph)
        self.create_model(graph, server_embedding_func)

    def create_model(self, graph: Graph, server_embedding_func):
        self.f_model = FGNN(graph)
        self.s_model = SGNNSlave(graph, server_embedding_func)

    def state_dict(self):
        weights = {}
        weights["fmodel"] = self.f_model.state_dict()
        return weights

    def load_state_dict(self, weights):
        self.f_model.load_state_dict(weights["fmodel"])


class FedGNNMaster(FedClassifier):
    def __init__(self, fgraph: Graph, sgraph: Graph):
        super().__init__(fgraph)
        self.create_model(fgraph, sgraph)

    def create_model(self, fgraph: Graph, sgraph: Graph):
        self.f_model = FGNN(fgraph)
        self.s_model = SGNNMaster(sgraph)

    def get_embeddings(self, node_ids=None):
        H = self.f_model()
        S = self.s_model(node_ids)
        O = H + S
        return O

    def get_embeddings_func(self):
        return self.s_model.get_embeddings

    def state_dict(self):
        weights = {}
        weights["fmodel"] = self.f_model.state_dict()
        return weights

    def load_state_dict(self, weights):
        self.f_model.load_state_dict(weights["fmodel"])


class FedDGCN(FedClassifier):
    def __init__(self, fgraph: AGraph, sgraph: AGraph):
        super().__init__(fgraph)
        self.create_model(fgraph, sgraph)

    def create_model(self, fgraph: AGraph, sgraph: AGraph):
        self.f_model = DGCN(fgraph)
        self.s_model = SDGCN(sgraph)
