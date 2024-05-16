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
        id: int = 0,
        classifier_type="GNN",
        save_path="./",
        logger=None,
    ):
        self.id = id
        self.graph = graph
        self.classifier_type = classifier_type
        self.save_path = save_path

        self.LOGGER = logger or logging

        # self.LOGGER.info(f"Client{self.id} statistics:")
        # self.LOGGER.info(f"Number of nodes: {self.graph.num_nodes}")

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

    def get_train_results(self, eval_=True):
        (
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_acc_f,
            val_acc_s,
        ) = self.train_step(eval_=eval_)

        result = {
            "Train Loss": round(train_loss.item(), 4),
            "Train Acc": round(train_acc, 4),
            "Val Loss": round(val_loss.item(), 4),
            "Val Acc": round(val_acc, 4),
            "Val F Acc": round(val_acc_f, 4),
            "Val S Acc": round(val_acc_s, 4),
        }

        return result

    def get_test_results(self):
        test_acc, test_acc_f, test_acc_s = self.test_classifier()

        if test_acc_s is not None:
            result = {
                "Test Acc": round(test_acc, 4),
                "Test Acc F": round(test_acc_f, 4),
                "Test Acc S": round(test_acc_s, 4),
            }
        else:
            result = {"Test Acc": round(test_acc, 4)}

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
            result = self.get_train_results(eval_=log)
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
        self.classifier.reset()

    def train_step(self, eval_=True):
        return self.classifier.train_step(eval_=eval_)
