from torch_geometric.loader import NeighborLoader

from src import *
from src.GNN.DGCN import DGCN, SDGCN, SDGCNMaster
from src.GNN.fGNN import FGNN
from src.GNN.sGNN import SGNNMaster, SGNNSlave
from src.utils.graph import AGraph, Graph
from src.utils.data import Data
from src.classifier import Classifier


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

    def train_step(self, eval_=True):
        res = super().train_step(eval_=eval_)

        if eval_:
            y = self.graph.y
            _, val_mask, _ = self.graph.get_masks()

            val_acc_f = Classifier.calc_metrics(self.f_model, y, val_mask, "acc")
            val_acc_s = Classifier.calc_metrics(self.s_model, y, val_mask, "acc")

            return (res[0], res[1], res[2], res[3], val_acc_f[0], val_acc_s[0])
        else:
            return res


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


class FedDGCNSlave(FedGNNSlave):
    def create_model(self, graph: AGraph, server_embedding_func):
        self.f_model = DGCN(graph)
        self.s_model = SGNNSlave(graph, server_embedding_func)


class FedDGCNMaster(FedGNNMaster):
    def create_model(self, fgraph: AGraph, sgraph: AGraph):
        self.f_model = DGCN(fgraph)
        self.s_model = SDGCNMaster(sgraph)
