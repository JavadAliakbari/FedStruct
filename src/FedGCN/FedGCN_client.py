from src import *
from src.FedGCN.FedGCN_classifier import FedGCNClassifier
from src.client import Client
from src.classifier import Classifier
from src.utils.graph import Graph


class FedGCNClient(Client):
    def __init__(self, graph: Graph, id: int = 0):
        super().__init__(graph=graph, id=id, classifier_type="GNN")
        # LOGGER.info(f"Number of edges: {self.graph.num_edges}")
        self.classifier: Classifier | None = None

    def initialize(self, **kwargs) -> None:
        self.classifier = FedGCNClassifier(self.graph)

    def train_local_model(
        self,
        epochs=config.model.iterations,
        log=True,
        plot=True,
        **kwargs,
    ):
        model_type = f"Server FedGCN"
        self.initialize()
        return super().train_local_model(
            epochs=epochs,
            log=log,
            plot=plot,
            model_type=model_type,
        )
