import os
import logging

from tqdm import tqdm
from src.classifier import Classifier

from src.utils.utils import *
from src.utils.config_parser import Config
from src.utils.graph import Graph

path = os.environ.get("CONFIG_PATH")
config = Config(path)


class Client:
    def __init__(
        self,
        graph: Graph,
        num_classes,
        id: int = 0,
        classifier_type="GNN",
        save_path="./",
        logger=None,
    ):
        self.id = id
        self.graph = graph
        self.num_classes = num_classes
        self.classifier_type = classifier_type
        self.save_path = save_path

        self.LOGGER = logger or logging

        self.LOGGER.info(f"Client{self.id} statistics:")
        self.LOGGER.info(f"Number of nodes: {self.graph.num_nodes}")
        self.LOGGER.info(f"Number of features: {self.graph.num_features}")

        self.classifier: Classifier = None

    def get_nodes(self):
        return self.graph.node_ids

    def num_nodes(self) -> int:
        return len(self.graph.node_ids)

    def parameters(self):
        return self.classifier.parameters()

    def zero_grad(self):
        self.classifier.zero_grad()

    def reset_parameters(self):
        self.classifier.reset_parameters()

    def state_dict(self):
        return self.classifier.state_dict()

    def load_state_dict(self, weights):
        self.classifier.load_state_dict(weights)

    def train(self, mode: bool = True):
        self.classifier.train(mode)

    def eval(self):
        self.classifier.eval()

    def test_classifier(self, metric=config.model.metric):
        return self.classifier.calc_test_accuracy(metric)

    def get_train_results(self, scale=False):
        (
            train_loss,
            train_acc,
            train_f1_score,
            val_loss,
            val_acc,
            val_f1_score,
        ) = self.train_step(scale)

        result = {
            "Train Loss": round(train_loss.item(), 4),
            "Train Acc": round(train_acc, 4),
            "Val Loss": round(val_loss.item(), 4),
            "Val Acc": round(val_acc, 4),
        }

        return result

    def get_test_results(self):
        test_acc = self.test_classifier()

        result = {
            "Test Acc": round(test_acc, 4),
        }

        return result

    def report_result(self, result, framework=""):
        self.LOGGER.info(f"{framework} results for client{self.id}:")
        self.LOGGER.info(f"{result}")

    def train_local_model(
        self,
        epochs=config.model.epoch_classifier,
        log=True,
        plot=True,
        model_type="GNN",
    ):
        self.LOGGER.info("local training starts!")

        if log:
            bar = tqdm(total=epochs, position=0)

        results = []
        for epoch in range(epochs):
            self.reset_model()

            self.train()
            result = self.get_train_results()
            result["Epoch"] = epoch + 1
            results.append(result)

            self.update_model()

            if log:
                bar.set_postfix(result)
                bar.update()

                if epoch == epochs - 1:
                    self.report_result(result, "Local Training")

        if plot:
            title = f"client {self.id} Local Training {model_type}"
            plot_metrics(results, title=title, save_path=self.save_path)

        test_results = self.get_test_results()
        for key, val in test_results.items():
            self.LOGGER.info(f"Client {self.id} {key}: {val:0.4f}")

        return test_results

    def get_grads(self, just_SFV=False):
        return self.classifier.get_model_grads(just_SFV)

    def set_grads(self, grads):
        self.classifier.set_model_grads(grads)

    def update_model(self):
        self.classifier.update_model()

    def reset_model(self):
        self.classifier.reset_classifier()

    def train_step(self, scale=False):
        return self.classifier.train_step(scale)
