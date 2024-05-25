import os

import torch

from src.models.model_binders import ModelBinder
from src.utils.config_parser import Config
from src.utils.graph import Data, Graph
from src.utils.utils import calc_metrics

path = os.environ.get("CONFIG_PATH")
config = Config(path)


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
            self.eval()
            y_pred_val = self.get_prediction()
            val_loss, val_acc = calc_metrics(y, y_pred_val, val_mask)

            return train_loss.item(), train_acc, val_loss.item(), val_acc
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
