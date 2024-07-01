import torch
from sklearn import model_selection
import torch.utils
import torch.utils.data

from src import *


class Data:
    def __init__(
        self,
        x: torch.Tensor = None,
        y: torch.Tensor = None,
        node_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        self.x = x
        self.y = y
        self.node_ids = node_ids
        self.num_nodes = node_ids.shape[0]
        self.num_features = x.shape[1]

        self.train_mask = kwargs.get("train_mask", None)
        self.test_mask = kwargs.get("test_mask", None)
        self.val_mask = kwargs.get("val_mask", None)
        self.num_classes = kwargs.get("num_classes", None)

    def get_masks(self):
        return (self.train_mask, self.val_mask, self.test_mask)

    def set_masks(self, masks):
        self.train_mask = masks[0]
        self.val_mask = masks[1]
        self.test_mask = masks[2]

    def add_masks(self, train_ratio=0.5, test_ratio=0.2):
        num_nodes = self.num_nodes
        indices = torch.arange(num_nodes, device=dev)
        train_size = int(train_ratio * num_nodes)
        test_size = int(test_ratio * num_nodes)
        val_size = num_nodes - train_size - test_size

        train_indices, val_indices, test_indices = torch.utils.data.random_split(
            indices,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed),
        )

        self.train_mask = indices.unsqueeze(1).eq(torch.tensor(train_indices)).any(1)
        self.val_mask = indices.unsqueeze(1).eq(torch.tensor(val_indices)).any(1)
        self.test_mask = indices.unsqueeze(1).eq(torch.tensor(test_indices)).any(1)
        # self.val_mask = ~(self.test_mask | self.train_mask)

        a = 2
