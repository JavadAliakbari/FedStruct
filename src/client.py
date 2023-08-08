import logging
import operator

from src.utils.config_parser import Config
from src.utils.graph import Graph
from src.GNN_classifier import GNNClassifier
from src.MLP_classifier import MLPClassifier
from src.models.GNN_models import GNN, MLP

config = Config()


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

        self.gnn_classifier = GNNClassifier(
            id=self.id,
            num_classes=self.num_classes,
            logger=self.LOGGER,
        )

        self.mlp_classifier = MLPClassifier(
            id=self.id,
            num_classes=self.num_classes,
            logger=self.LOGGER,
        )

    def get_nodes(self):
        return self.subgraph.node_ids

    def create_local_sd_model(self, sd_dims):
        cls_dims = [self.subgraph.num_features] + config.model.classifier_layer_sizes
        in_dims = list(map(operator.add, cls_dims, sd_dims))
        out_dims = config.model.classifier_layer_sizes + [self.num_classes]
        self.local_sd_model = GNN(
            in_dims=in_dims,
            out_dims=out_dims,
            linear_layer=True,
            dropout=config.model.dropout,
            last_layer="softmax",
        )
        # self.local_sd_model = MLP(
        #     layer_sizes=cls_dims,
        #     dropout=config.dropout,
        #     softmax=False,
        # )

    def get_local_sd_model(self):
        return self.local_sd_model

    def reset_gnn_parameters(self):
        self.gnn_classifier.reset_parameters()

    def get_gnn_weights(self):
        return self.gnn_classifier.state_dict()

    def set_gnn_weights(self, weights):
        self.gnn_classifier.load_state_dict(weights)

    def initialize_gnn_(
        graph: Graph,
        classifier: GNNClassifier,
        dim_in=None,
    ) -> None:
        classifier.prepare_data(
            graph=graph,
            batch_size=config.model.batch_size,
            num_neighbors=config.model.num_samples,
            shuffle=True,
        )

        classifier.set_classifiers(dim_in=dim_in)

    def initialize_gnn(self, input_dimension=None) -> None:
        Client.initialize_gnn_(
            self.subgraph, self.gnn_classifier, dim_in=input_dimension
        )

    def fit_gnn(self, epochs, bar=False, plot=False, type="local") -> None:
        return self.gnn_classifier.fit(
            epochs=epochs,
            batch=False,
            bar=bar,
            plot=plot,
            type=type,
        )

    def train_local_gnn(self) -> None:
        self.initialize_gnn()
        return self.fit_gnn(
            config.model.epoch_classifier,
            bar=True,
            plot=True,
            type="local",
        )

    def test_local_gnn(self):
        return self.gnn_classifier.calc_test_accuracy()

    def reset_mlp_parameters(self):
        self.mlp_classifier.reset_parameters()

    def get_mlp_weights(self):
        return self.mlp_classifier.state_dict()

    def set_mlp_weights(self, weights):
        self.mlp_classifier.load_state_dict(weights)

    def initialize_mlp_(
        graph: Graph,
        classifier: MLPClassifier,
        dim_in=None,
    ) -> None:
        classifier.prepare_data(
            graph=graph,
            batch_size=config.model.batch_size,
        )

        classifier.set_classifiers(dim_in=dim_in)

    def initialize_mlp(self, input_dimension=None) -> None:
        Client.initialize_mlp_(
            self.subgraph, self.mlp_classifier, dim_in=input_dimension
        )

    def fit_mlp(self, epochs, bar=False, plot=False, type="local") -> None:
        return self.mlp_classifier.fit(
            epochs=epochs,
            # batch=False,
            bar=bar,
            plot=plot,
            type=type,
        )

    def train_local_mlp(self) -> None:
        self.initialize_mlp()
        return self.fit_mlp(
            config.model.epoch_classifier,
            bar=True,
            plot=True,
            type="local",
        )

    def test_local_mlp(self):
        return self.mlp_classifier.calc_test_accuracy()
