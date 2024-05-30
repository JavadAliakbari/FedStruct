import torch
from torch.utils.data import DataLoader

from src import *
from src.utils.data import Data
from src.classifier import Classifier
from src.models.model_binders import ModelBinder, ModelSpecs
from src.utils.utils import calc_metrics, calc_accuracy, calc_f1_score


class MLPClassifier(Classifier):
    def __init__(self, graph: Data):
        super().__init__()
        self.prepare_data(graph)
        self.create_model()

    def create_model(self):
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

        train_x = self.graph.x[self.graph.train_mask]
        train_y = self.graph.y[self.graph.train_mask]
        val_x = self.graph.x[self.graph.val_mask]
        val_y = self.graph.y[self.graph.val_mask]
        test_x = self.graph.x[self.graph.test_mask]
        test_y = self.graph.y[self.graph.test_mask]

        self.train_loader = DataLoader(
            list(zip(train_x, train_y)), batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            list(zip(val_x, val_y)), batch_size=batch_size, shuffle=False
        )

        self.test_data = [test_x, test_y]

    def get_embeddings(self):
        H = self.model(self.graph.x)
        return H

    def get_prediction(self):
        H = self.get_embeddings()
        y_pred = torch.nn.functional.softmax(H, dim=1)
        return y_pred

    def train_step(self, eval_=True):
        y_pred = self.get_prediction()
        y = self.graph.y

        train_mask, val_mask, _ = self.graph.get_masks()

        train_loss, train_acc = calc_metrics(y, y_pred, train_mask)
        train_loss.backward(retain_graph=False)
        if eval_:
            self.eval()
            val_loss, val_acc = calc_metrics(y, y_pred, val_mask)

            return train_loss.item(), train_acc, val_loss.item(), val_acc
        else:
            return train_loss.item(), train_acc, 0, 0

    @torch.no_grad()
    def calc_test_accuracy(self, metric="acc"):
        self.model.eval()
        test_x, test_y = self.test_data
        out = self.model(test_x)
        if metric == "acc":
            val_acc = calc_accuracy(out.argmax(dim=1), test_y)
        elif metric == "f1":
            val_acc = calc_f1_score(out.argmax(dim=1), test_y)

        return (val_acc,)
