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
        classifier_type="GNN",
        logger=None,
    ):
        self.id = id
        self.subgraph = subgraph_
        self.num_classes = num_classes
        self.classifier_type = classifier_type

        self.LOGGER = logger or logging

        self.LOGGER.info(f"Client{self.id} statistics:")
        self.LOGGER.info(f"Number of nodes: {self.subgraph.num_nodes}")
        self.LOGGER.info(f"Number of edges: {self.subgraph.num_edges}")
        self.LOGGER.info(f"Number of features: {self.subgraph.num_features}")

        if classifier_type == "GNN":
            self.classifier = GNNClassifier(
                id=self.id,
                num_classes=self.num_classes,
                logger=self.LOGGER,
            )
        elif classifier_type == "MLP":
            self.classifier = MLPClassifier(
                id=self.id,
                num_classes=self.num_classes,
                logger=self.LOGGER,
            )

    def get_nodes(self):
        return self.subgraph.node_ids

    def num_nodes(self) -> int:
        return len(self.subgraph.node_ids)

    def create_local_sd_model(self, sd_dims, model_type="GNN"):
        self.model_type = model_type
        cls_dims = [self.subgraph.num_features] + config.model.classifier_layer_sizes
        # in_dims = list(map(operator.add, cls_dims, sd_dims))
        # out_dims = config.model.classifier_layer_sizes + [self.num_classes]
        # self.local_sd_model = GNN(
        #     in_dims=in_dims,
        #     out_dims=out_dims,
        #     linear_layer=True,
        #     dropout=config.model.dropout,
        #     last_layer="softmax",
        # )

        if self.model_type == "GNN":
            self.local_sd_model = GNN(
                in_dims=cls_dims,
                linear_layer=True,
                dropout=config.model.dropout,
                last_layer="linear",
                batch_normalization=True,
            )
        elif self.model_type == "MLP":
            self.local_sd_model = MLP(
                layer_sizes=cls_dims,
                dropout=config.model.dropout,
                softmax=False,
                batch_normalization=True,
            )

        decision_layer_sizes = [
            config.model.classifier_layer_sizes[-1] + sd_dims[-1],
            # 64,
            self.num_classes,
        ]
        self.local_dense_model = MLP(
            layer_sizes=decision_layer_sizes,
            dropout=config.model.dropout,
            softmax=True,
            batch_normalization=False,
        )

    def get_local_sd_model(self):
        return self.local_sd_model

    def get_sd_parameters(self):
        return list(self.local_sd_model.parameters()) + list(
            self.local_dense_model.parameters()
        )

    def reset_sd_parameters(self):
        self.local_sd_model.reset_parameters()
        self.local_dense_model.reset_parameters()

    def zero_grad_sd(self):
        self.local_sd_model.zero_grad()
        self.local_dense_model.zero_grad()

    def get_sd_weights(self):
        weights = {}
        weights["sd_model"] = self.local_sd_model.state_dict()
        weights["dense_model"] = self.local_dense_model.state_dict()

        return weights

    def set_sd_weights(self, weights):
        self.local_sd_model.load_state_dict(weights["sd_model"])
        self.local_dense_model.load_state_dict(weights["dense_model"])

    def get_feature_embeddings(self):
        x = self.subgraph.x
        if self.model_type == "GNN":
            edge_index = self.subgraph.edge_index
            h = self.local_sd_model(x, edge_index)
        elif self.model_type == "MLP":
            h = self.local_sd_model(x)
        return h

    def get_sd_labels(self, x):
        h = self.local_dense_model(x)
        return h

    def reset_classifier_parameters(self):
        self.classifier.reset_parameters()

    def get_classifier_weights(self):
        return self.classifier.state_dict()

    def set_classifier_weights(self, weights):
        self.classifier.load_state_dict(weights)

    def initialize_gnn_(
        graph: Graph,
        classifier: GNNClassifier,
        dim_in=None,
    ) -> None:
        classifier.prepare_data(
            graph=graph,
            batch_size=config.model.batch_size,
            num_neighbors=config.model.num_samples,
        )
        classifier.set_classifiers(dim_in=dim_in)

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

    def initialize_classifier(self, input_dimension=None) -> None:
        if self.classifier_type == "GNN":
            Client.initialize_gnn_(self.subgraph, self.classifier, input_dimension)
        elif self.classifier_type == "MLP":
            Client.initialize_mlp_(self.subgraph, self.classifier, input_dimension)

    def fit_classifier(self, epochs, bar=False, plot=False, type="local") -> None:
        return self.classifier.fit(
            epochs=epochs,
            batch=False,
            bar=bar,
            plot=plot,
            type=type,
        )

    def train_local_classifier(self, epochs, bar=True, plot=True) -> None:
        self.initialize_classifier()
        return self.fit_classifier(
            epochs,
            bar=bar,
            plot=plot,
            type="local",
        )

    def test_local_classifier(self):
        return self.classifier.calc_test_accuracy()
