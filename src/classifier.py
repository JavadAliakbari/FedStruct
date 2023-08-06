from ast import List
import logging
import os
from typing import Union

import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.loader import NeighborLoader

from src.models.GNN_models import GNN, calc_accuracy, test, calc_f1_score
from src.utils.graph import Graph
from src.utils import config


class Classifier:
    def __init__(
        self,
        id,
        num_classes,
        logger=None,
    ):
        self.id = id
        self.num_classes = num_classes

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

    def plot_results(res, client_id, type="local", model_type="GNN"):
        dataset = pd.DataFrame.from_dict(res)
        dataset.set_index("Epoch", inplace=True)

        save_dir = f"./plot_results/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        loss_columns = list(filter(lambda x: x.endswith("Loss"), dataset.columns))
        dataset[loss_columns].plot()
        title = f"{type} {model_type} loss client {client_id}"
        plt.title(title)
        plt.savefig(f"{save_dir}{title}.png")

        acc_columns = list(filter(lambda x: x.endswith("Acc"), dataset.columns))
        dataset[acc_columns].plot()
        title = f"{type} {model_type} accuracy client {client_id}"
        plt.title(title)

        plt.savefig(f"{save_dir}{title}.png")
