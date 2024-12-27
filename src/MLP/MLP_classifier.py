import torch
from torch.utils.data import DataLoader

from src import *
from src.utils.data import Data
from src.classifier import Classifier
from src.models.model_binders import ModelBinder, ModelSpecs


class MLPClassifier(Classifier):
    def __init__(self, graph: Data):
        super().__init__()
        self.prepare_data(graph)
        self.create_smodel()

    def create_smodel(self):
        layer_sizes = (
            [self.graph.num_features]
            + config.feature_model.mlp_layer_sizes
            + [self.graph.num_classes]
        )
        model_specs = [
            ModelSpecs(
                type="MLP",
                layer_sizes=layer_sizes,
                final_activation_function="linear",
                normalization="layer",
            ),
        ]
        self.model: ModelBinder = ModelBinder(model_specs)
        self.model.to(device)

    def prepare_data(
        self,
        data: Data,
        batch_size: int = config.model.batch_size,
    ):
        self.graph = data
        if self.graph.train_mask is None:
            self.graph.add_masks()

        # train_x = self.graph.x[self.graph.train_mask]
        # train_y = self.graph.y[self.graph.train_mask]
        # val_x = self.graph.x[self.graph.val_mask]
        # val_y = self.graph.y[self.graph.val_mask]
        # test_x = self.graph.x[self.graph.test_mask]
        # test_y = self.graph.y[self.graph.test_mask]

        # self.train_loader = DataLoader(
        #     list(zip(train_x, train_y)), batch_size=batch_size, shuffle=True
        # )
        # self.val_loader = DataLoader(
        #     list(zip(val_x, val_y)), batch_size=batch_size, shuffle=False
        # )

        # self.test_data = [test_x, test_y]

    def get_embeddings(self):
        H = self.model(self.graph.x)
        return H
