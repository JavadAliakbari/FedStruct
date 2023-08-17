import logging

from src.utils.config_parser import Config
from src.utils.graph import Graph
from src.GNN_classifier import GNNClassifier
from src.MLP_classifier import MLPClassifier

config = Config()


class Client:
    def __init__(
        self,
        subgraph_: Graph,
        num_classes,
        id: int = 0,
        classifier_type="GNN",
        save_path="./",
        logger=None,
    ):
        self.id = id
        self.subgraph = subgraph_
        self.num_classes = num_classes
        self.classifier_type = classifier_type
        self.save_path = save_path

        self.LOGGER = logger or logging

        self.LOGGER.info(f"Client{self.id} statistics:")
        self.LOGGER.info(f"Number of nodes: {self.subgraph.num_nodes}")
        self.LOGGER.info(f"Number of edges: {self.subgraph.num_edges}")
        self.LOGGER.info(f"Number of features: {self.subgraph.num_features}")

        if classifier_type == "GNN":
            self.classifier = GNNClassifier(
                id=self.id,
                num_classes=self.num_classes,
                save_path=self.save_path,
                logger=self.LOGGER,
            )
        elif classifier_type == "MLP":
            self.classifier = MLPClassifier(
                id=self.id,
                num_classes=self.num_classes,
                save_path=self.save_path,
                logger=self.LOGGER,
            )

    def get_nodes(self):
        return self.subgraph.node_ids

    def num_nodes(self) -> int:
        return len(self.subgraph.node_ids)

    def parameters(self):
        return self.classifier.parameters()

    def zero_grad(self):
        self.classifier.zero_grad()

    def get_feature_embeddings(self):
        return self.classifier.get_feature_embeddings()

    def predict_labels(self, x):
        return self.classifier.predict_labels(x)

    def reset_parameters(self):
        self.classifier.reset_parameters()

    def state_dict(self):
        return self.classifier.state_dict()

    def load_state_dict(self, weights):
        self.classifier.load_state_dict(weights)

    def initialize_gnn_(
        graph: Graph, classifier: GNNClassifier, dim_in=None, additional_layer_dims=0
    ) -> None:
        classifier.prepare_data(
            graph=graph,
            batch_size=config.model.batch_size,
            num_neighbors=config.model.num_samples,
        )
        classifier.set_classifiers(
            dim_in=dim_in, additional_layer_dims=additional_layer_dims
        )

    def initialize_mlp_(
        graph: Graph,
        classifier: MLPClassifier,
        dim_in=None,
    ):
        classifier.prepare_data(
            graph=graph,
            batch_size=config.model.batch_size,
        )
        classifier.set_classifiers(dim_in=dim_in)

    def initialize(self, input_dimension=None, additional_layer_dims=0) -> None:
        if self.classifier_type == "GNN":
            Client.initialize_gnn_(
                self.subgraph, self.classifier, input_dimension, additional_layer_dims
            )
        elif self.classifier_type == "MLP":
            Client.initialize_mlp_(self.subgraph, self.classifier, input_dimension)

    def fit(self, epochs, bar=False, plot=False, type="local") -> None:
        return self.classifier.fit(
            epochs=epochs,
            batch=False,
            bar=bar,
            plot=plot,
            type=type,
        )

    def train_local_classifier(self, epochs, bar=True, plot=True) -> None:
        self.initialize()
        return self.fit(
            epochs,
            bar=bar,
            plot=plot,
            type="local",
        )

    def test_local_classifier(self):
        return self.classifier.calc_test_accuracy()
