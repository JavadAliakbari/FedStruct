import logging
from typing import Union

import torch


class Classifier:
    def __init__(
        self,
        id,
        num_classes,
        save_path="./",
        logger=None,
    ):
        self.id = id
        self.num_classes = num_classes
        self.save_path = save_path

        self.LOGGER = logger or logging

        self.model: Union[None, torch.nn.Module] = None

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, weights):
        self.model.load_state_dict(weights)

    def reset_parameters(self):
        self.model.reset_parameters()

    def parameters(self):
        return self.model.parameters()

    def train(self, mode: bool = True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()

    def get_model_grads(self):
        client_parameters = list(self.parameters())
        client_grads = [client_parameter.grad for client_parameter in client_parameters]

        return client_grads

    def zero_grad(self, set_to_none=False):
        self.model.zero_grad(set_to_none=set_to_none)
