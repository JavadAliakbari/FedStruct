from src import *
from src.utils.data import Data
from src.client import Client
from src.MLP.MLP_classifier import MLPClassifier


class MLPClient(Client):
    def __init__(self, graph: Data, id: int = 0):
        super().__init__(graph=graph, id=id, classifier_type="MLP")

    def initialize(self):
        self.classifier = MLPClassifier(self.graph)
        self.classifier.create_optimizer()

    def train_local_model(
        self,
        epochs=config.model.iterations,
        log=True,
        plot=True,
    ):
        self.initialize()
        return super().train_local_model(
            epochs=epochs,
            log=log,
            plot=plot,
            model_type="MLP",
        )
