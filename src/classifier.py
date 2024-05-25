import os

import torch

from src.models.model_binders import ModelBinder
from src.utils.config_parser import Config
from src.utils.graph import Data, Graph

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

    def get_embeddings(self):
        raise NotImplementedError

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

    def get_embeddings_func(self):
        return self.get_embeddings
