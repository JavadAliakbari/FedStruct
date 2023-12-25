import os

import numpy as np

from src.GNN_client import GNNClient
from src.utils.utils import *
from src.utils.config_parser import Config
from src.utils.graph import Graph
from src.neighgen import NeighGen
from src.models.feature_loss import greedy_loss

path = os.environ.get("CONFIG_PATH")
config = Config(path)


class FedSAGEClient(GNNClient):
    def __init__(
        self,
        graph: Graph,
        num_classes,
        id: int = 0,
        save_path="./",
        logger=None,
    ):
        super().__init__(
            graph=graph,
            num_classes=num_classes,
            id=id,
            save_path=save_path,
            logger=logger,
        )

        self.neighgen = NeighGen(
            id=self.id,
            num_classes=self.num_classes,
            save_path=self.save_path,
            logger=self.LOGGER,
        )

    def initialize(
        self,
        propagate_type=config.model.propagate_type,
        num_input_features=None,
        **kwargs,
    ) -> None:
        mend_graph = self.neighgen.get_mend_graph()
        if mend_graph is not None:
            self.classifier.prepare_data(
                graph=mend_graph,
                batch_size=config.model.batch_size,
                num_neighbors=config.model.num_samples,
            )
        else:
            self.classifier.prepare_data(
                graph=self.graph,
                batch_size=config.model.batch_size,
                num_neighbors=config.model.num_samples,
            )

        # self.classifier.set_GNN_FPM(dim_in=num_input_features)
        if propagate_type == "GNN":
            self.classifier.set_GNN_FPM(dim_in=num_input_features)
        elif propagate_type == "MP":
            self.classifier.set_DGCN_FPM(dim_in=num_input_features)

    def reset_neighgen_parameters(self):
        self.neighgen.reset_parameters()

    def get_neighgen_weights(self):
        return self.neighgen.state_dict()

    def set_neighgen_weights(self, weights):
        self.neighgen.load_state_dict(weights)

    def initialize_neighgen(self) -> None:
        self.neighgen.prepare_data(self.graph)
        self.neighgen.set_model()

    def predict_neighgen(self):
        return self.neighgen.predict_missing_neigh()

    def calc_loss_neighgen(self, pred_missing, pred_feat, pred_label, mask):
        return self.neighgen.calc_loss(pred_missing, pred_feat, pred_label, mask)

    def calc_accuracy_neighgen(self, pred_label, pred_feat, mask):
        self.neighgen.calc_accuracy(pred_label, pred_feat, mask)

    def create_inter_features(self, pred_missing, pred_features, true_missing, mask):
        num_train_nodes = sum(mask.numpy())
        all_nodes = self.graph.node_ids.numpy()
        selected_node_ids = np.random.choice(all_nodes, num_train_nodes)
        # remaining_nodes: list = np.setdiff1d(all_nodes, selected_node_ids)
        # np.random.shuffle(remaining_nodes)
        # remaining_nodes = remaining_nodes.tolist()

        inter_features = []
        for node_id in selected_node_ids:
            neighbors_ids = self.graph.find_neigbors(node_id)
            while len(neighbors_ids) == 0:
                np.random.shuffle(all_nodes)
                replace_node_id = all_nodes[0]
                neighbors_ids = self.graph.find_neigbors(replace_node_id)
            selected_neighbors_ids = np.random.choice(
                neighbors_ids,
                config.fedsage.num_pred,
            )
            inter_features.append(
                self.graph.x[np.isin(self.graph.node_ids, selected_neighbors_ids)]
            )

        # inter_features = torch.tensor(np.array(inter_features))
        inter_loss = greedy_loss(
            pred_features[mask],
            inter_features,
            pred_missing[mask],
        )

        return inter_loss

    def train_neighgen(
        self,
        inter_client_features_creators: list = [],
        log=True,
        plot=True,
    ) -> None:
        self.initialize_neighgen()
        self.neighgen.fit(
            config.fedsage.neighgen_epochs,
            inter_client_features_creators,
            log=log,
            plot=plot,
        )

    def create_mend_graph(self):
        self.neighgen.create_mend_graph()

    def train_locsage(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        inter_client_features_creators: list = [],
        log=False,
        plot=False,
    ):
        # self.initialize_locsage(inter_client_features_creators)
        self.train_neighgen(inter_client_features_creators)
        self.create_mend_graph()
        self.train_local_model(
            epochs=epochs,
            propagate_type=propagate_type,
            log=log,
            plot=plot,
        )
