import logging
from typing import Union

import torch

from src.models.GNN_models import ModelBinder


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

        self.feature_model: Union[None, ModelBinder] = None
        self.structure_model: Union[None, ModelBinder] = None
        self.SFV: Union[None, torch.Tensor] = None
        self.optimizer = None

    def state_dict(self):
        weights = {}
        weights["FPM"] = self.feature_model.state_dict()
        if self.structure_model is not None:
            weights["SPM"] = self.structure_model.state_dict()

        # if self.SFV is not None:
        #     if self.SFV.requires_grad:
        #         self.SFV.grad = None
        #         weights["SFV"] = self.SFV

        return weights

    def load_state_dict(self, weights):
        self.feature_model.load_state_dict(weights["FPM"])
        if self.structure_model is not None and "SPM" in weights.keys():
            self.structure_model.load_state_dict(weights["SPM"])

        # if "SFV" in weights.keys():
        #     self.SFV = deepcopy(weights["SFV"])

    def get_model_grads(self, just_SFV=False):
        grads = {}
        if not just_SFV:
            grads["FPM"] = self.feature_model.get_grads()
            if self.structure_model is not None:
                grads["SPM"] = self.structure_model.get_grads()

        if self.SFV is not None:
            if self.SFV.requires_grad:
                grads["SFV"] = [self.SFV.grad]

        return grads

    def set_model_grads(self, grads):
        if "FPM" in grads.keys():
            self.feature_model.set_grads(grads["FPM"])
        if "SPM" in grads.keys():
            self.structure_model.set_grads(grads["SPM"])
        if "SFV" in grads.keys():
            self.SFV.grad = grads["SFV"][0]

    # def get_SFV_grad(self):
    #     return self.SFV.grad

    # def set_SFV_grad(self, grad):
    #     self.SFV.grad = grad

    def reset_parameters(self):
        self.feature_model.reset_parameters()
        if self.structure_model is not None:
            self.structure_model.reset_parameters()

    def parameters(self):
        parameters = self.feature_model.parameters()
        if self.structure_model is not None:
            parameters += self.structure_model.parameters()

        if self.SFV is not None:
            if self.SFV.requires_grad:
                parameters += [self.SFV]

        return parameters

    def train(self, mode: bool = True):
        self.feature_model.train(mode)
        if self.structure_model is not None:
            self.structure_model.train(mode)

    def eval(self):
        self.feature_model.eval()
        if self.structure_model is not None:
            self.structure_model.eval()

    def zero_grad(self, set_to_none=False):
        self.feature_model.zero_grad(set_to_none=set_to_none)
        if self.structure_model is not None:
            self.structure_model.zero_grad(set_to_none=set_to_none)

        if self.SFV is not None:
            if self.SFV.requires_grad:
                self.SFV.grad = None

    def update_model(self):
        self.optimizer.step()

    def reset_classifier(self):
        self.optimizer.zero_grad()
