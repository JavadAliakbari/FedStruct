import logging
import numpy as np

from src.utils import config
from src.utils.graph import Graph
from src.classifier import Classifier
from src.models.feature_loss import greedy_loss


class Client:
    def __init__(
        self,
        subgraph_: Graph,
        num_classes,
        id: int = 0,
        logger=None,
    ):
        self.id = id
        self.subgraph = subgraph_
        self.num_classes = num_classes

        self.LOGGER = logger or logging

        self.LOGGER.info(f"Client{self.id} statistics:")
        self.LOGGER.info(f"Number of nodes: {self.subgraph.num_nodes}")
        self.LOGGER.info(f"Number of edges: {self.subgraph.num_edges}")
        self.LOGGER.info(f"Number of features: {self.subgraph.num_features}")

        self.classifier = Classifier(
            id=self.id,
            num_classes=self.num_classes,
            logger=self.LOGGER,
        )

    def get_nodes(self):
        return self.subgraph.node_ids

    def reset_classifier_parameters(self):
        self.classifier.reset_parameters()

    def get_classifier_weights(self):
        return self.classifier.state_dict()

    def set_classifier_weights(self, weights):
        self.classifier.load_state_dict(weights)

    def initialize_classifier_(
        graph: Graph,
        classifier: Classifier,
        dim_in=None,
    ) -> None:
        classifier.prepare_data(
            graph=graph,
            batch_size=config.batch_size,
            num_neighbors=config.num_samples,
            shuffle=True,
        )

        classifier.set_classifiers(dim_in=dim_in)

    def initialize_classifier(self, input_dimension=None) -> None:
        Client.initialize_classifier_(
            self.subgraph, self.classifier, dim_in=input_dimension
        )

    def fit_classifier(self, epochs, bar=False, plot=False, type="local") -> None:
        return self.classifier.fit(
            epochs=epochs,
            batch=True,
            bar=bar,
            plot=plot,
            type=type,
        )

    def train_local_classifier(self) -> None:
        self.initialize_classifier()
        return self.fit_classifier(
            config.epoch_classifier,
            bar=True,
            plot=True,
            type="local",
        )

    def test_local_classifier(self):
        return self.classifier.calc_test_accuracy()

    def joint_train_classifier(
        self,
        structure_features,
        structure_loss,
        retain_graph=False,
    ):
        return self.classifier.joint_train(
            structure_features,
            structure_loss,
            retain_graph=retain_graph,
        )
