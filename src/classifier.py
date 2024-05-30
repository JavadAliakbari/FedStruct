import torch

from src import *
from src.models.model_binders import ModelBinder
from src.utils.graph import Graph
from src.utils.data import Data


class Classifier:
    def __init__(self):
        self.graph: Graph | Data | None = None
        self.model: ModelBinder | None = None
        self.optimizer = None

    def create_model(self):
        raise NotImplementedError

    def create_optimizer(self):
        parameters = self.parameters()
        if len(parameters) == 0:
            return
        self.optimizer = torch.optim.Adam(
            parameters,
            lr=config.model.lr,
            weight_decay=config.model.weight_decay,
        )

    def state_dict(self):
        weights = {}
        if self.model is not None:
            weights["model"] = self.model.state_dict()

        return weights

    def load_state_dict(self, weights):
        if self.model is not None:
            self.model.load_state_dict(weights["model"])

    def get_grads(self, just_SFV=False):
        if just_SFV:
            return {}
        grads = {}
        if self.model is not None:
            grads["model"] = self.model.get_grads()

        return grads

    def set_grads(self, grads):
        if "model" in grads.keys():
            self.model.set_grads(grads["model"])

    def reset_parameters(self):
        if self.model is not None:
            self.model.reset_parameters()

    def parameters(self):
        parameters = []
        if self.model is not None:
            parameters += self.model.parameters()

        return parameters

    def train(self, mode: bool = True):
        if self.model is not None:
            self.model.train(mode)

    def eval(self):
        if self.model is not None:
            self.model.eval()

    def zero_grad(self, set_to_none=False):
        if self.model is not None:
            self.model.zero_grad(set_to_none=set_to_none)

    def update_model(self):
        if self.optimizer is not None:
            self.optimizer.step()

    def reset(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def restart(self):
        self.graph = None
        self.model = None
        self.optimizer = None

    def get_embeddings(self):
        raise NotImplementedError

    def get_embeddings_func(self):
        return self.get_embeddings

    def __call__(self):
        return self.get_embeddings()

    def get_prediction(self):
        H = self.get_embeddings()
        y_pred = torch.nn.functional.softmax(H, dim=1)
        return y_pred

    def train_step(self, eval_=True):
        y_pred = self.get_prediction()
        y = self.graph.y
        train_mask, val_mask, _ = self.graph.get_masks()

        train_loss, train_acc = calc_metrics(y, y_pred, train_mask)
        train_loss.backward(retain_graph=True)

        if eval_:
            val_loss, val_acc = Classifier.calc_metrics(self, y, val_mask)
            return train_loss.item(), train_acc, val_loss.item(), val_acc
        else:
            return train_loss.item(), train_acc

    def calc_test_accuracy(self, metric="acc"):
        return Classifier.calc_metrics(self, self.graph.y, self.graph.test_mask, metric)

    @torch.no_grad()
    def calc_metrics(model, y, mask, metric=""):
        model.eval()
        y_pred = model.get_prediction()
        loss, acc = calc_metrics(y, y_pred, mask)

        if metric == "acc":
            return (acc,)
        # elif metric == "f1":
        #     return f1_score
        elif metric == "loss":
            return (loss.item(),)
        else:
            return loss, acc
