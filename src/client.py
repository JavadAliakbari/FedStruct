import logging

import numpy as np

from src.utils.config_parser import Config
from src.utils.graph import Graph
from src.GNN_classifier import GNNClassifier
from src.MLP_classifier import MLPClassifier
from src.neighgen import NeighGen
from src.models.feature_loss import greedy_loss

config = Config()
config.num_pred = 5


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

        self.neighgen = NeighGen(
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

    def train(self, mode: bool = True):
        self.classifier.train(mode)

    def eval(self):
        self.classifier.eval()

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
            batch=config.model.batch,
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

    def reset_neighgen_parameters(self):
        self.neighgen.reset_parameters()

    def get_neighgen_weights(self):
        return self.neighgen.state_dict()

    def set_neighgen_weights(self, weights):
        self.neighgen.load_state_dict(weights)

    def initialize_neighgen(self) -> None:
        self.neighgen.prepare_data(
            x=self.subgraph.x,
            y=self.subgraph.y,
            edges=self.subgraph.get_edges(),
            node_ids=self.subgraph.node_ids,
        )
        self.neighgen.set_model()

    def predict_neighgen(self):
        return self.neighgen.predict_missing_neigh()

    def calc_loss_neighgen(self, pred_missing, pred_feat, pred_label, mask):
        return self.neighgen.calc_loss(pred_missing, pred_feat, pred_label, mask)

    def calc_accuracy_neighgen(self, pred_label, pred_feat, mask):
        self.neighgen.calc_accuracy(pred_label, pred_feat, mask)

    def create_inter_features(self, pred_missing, pred_features, true_missing, mask):
        num_train_nodes = sum(mask.numpy())
        all_nodes = self.subgraph.node_ids.numpy()
        selected_node_ids = np.random.choice(all_nodes, num_train_nodes)
        # remaining_nodes: list = np.setdiff1d(all_nodes, selected_node_ids)
        # np.random.shuffle(remaining_nodes)
        # remaining_nodes = remaining_nodes.tolist()

        inter_features = []
        for node_id in selected_node_ids:
            neighbors_ids = self.subgraph.find_neigbors(node_id)
            while len(neighbors_ids) == 0:
                np.random.shuffle(all_nodes)
                replace_node_id = all_nodes[0]
                neighbors_ids = self.subgraph.find_neigbors(replace_node_id)
            selected_neighbors_ids = np.random.choice(neighbors_ids, config.num_pred)
            inter_features.append(
                self.subgraph.x[np.isin(self.subgraph.node_ids, selected_neighbors_ids)]
            )

        # inter_features = torch.tensor(np.array(inter_features))
        inter_loss = greedy_loss(
            pred_features[mask],
            inter_features,
            pred_missing[mask],
        )

        return inter_loss

    def fit_neighgen(self, epochs, inter_client_features_creators: list = []) -> None:
        self.neighgen.fit(epochs, inter_client_features_creators)

    def train_neighgen(self, inter_client_features_creators: list = []) -> None:
        self.initialize_neighgen()
        self.fit_neighgen(config.model.gen_epochs, inter_client_features_creators)

    def initialize_locsage(self, inter_client_features_creators: list = []):
        self.train_neighgen(inter_client_features_creators)
        mend_graph = self.neighgen.create_mend_graph(self.subgraph)

        Client.initialize_gnn_(graph=mend_graph, classifier=self.classifier)

    def train_locsage(
        self,
        inter_client_features_creators: list = [],
        bar=False,
        plot=False,
    ):
        self.initialize_locsage(inter_client_features_creators)

        return self.classifier.fit(
            epochs=config.model.epoch_classifier,
            bar=bar,
            plot=plot,
            type="locsage",
        )
