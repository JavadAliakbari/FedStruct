import torch
from sklearn import model_selection

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

    def add_masks(self, train_size=0.5, test_size=0.2):
        num_nodes = self.num_nodes
        indices = torch.arange(num_nodes, device=dev)

        train_indices, test_indices = model_selection.train_test_split(
            indices,
            train_size=train_size,
            test_size=test_size,
        )

        self.train_mask = indices.unsqueeze(1).eq(train_indices).any(1)
        self.test_mask = indices.unsqueeze(1).eq(test_indices).any(1)
        self.val_mask = ~(self.test_mask | self.train_mask)
