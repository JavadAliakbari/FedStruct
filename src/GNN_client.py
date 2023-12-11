import numpy as np

from src.utils.utils import *
from src.utils.config_parser import Config
from src.client import Client
from src.utils.graph import Graph
from src.GNN_classifier import GNNClassifier
from src.neighgen import NeighGen
from src.models.feature_loss import greedy_loss

config = Config()
config.num_pred = 5


class GNNClient(Client):
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
            classifier_type="GNN",
            save_path=save_path,
            logger=logger,
        )

        self.LOGGER.info(f"Number of edges: {self.graph.num_edges}")

        self.classifier: GNNClassifier = GNNClassifier(
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

    def get_structure_embeddings2(self, node_ids):
        return self.classifier.get_structure_embeddings2(node_ids)

    def initialize(
        self,
        propagate_type=config.model.propagate_type,
        num_input_features=None,
        structure=False,
        get_structure_embeddings=None,
    ) -> None:
        self.classifier.prepare_data(
            graph=self.graph,
            batch_size=config.model.batch_size,
            num_neighbors=config.model.num_samples,
        )

        if propagate_type == "GNN":
            self.classifier.set_GNN_FPM(dim_in=num_input_features)
        elif propagate_type == "MP":
            self.classifier.set_DGCN_FPM(dim_in=num_input_features)

        if structure:
            if config.structure_model.structure_type != "GDV":
                dim_in = config.structure_model.num_structural_features
            else:
                dim_in = 73

            if propagate_type == "GNN":
                self.classifier.set_GNN_SPM(
                    dim_in=dim_in,
                    get_structure_embeddings=get_structure_embeddings,
                )
            elif propagate_type == "MP":
                self.classifier.set_DGCN_SPM(dim_in=dim_in)

    def train_local_model(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        log=True,
        plot=True,
        structure=False,
    ):
        self.initialize(propagate_type=propagate_type, structure=structure)
        return super().train_local_model(
            epochs=epochs,
            log=log,
            plot=plot,
            model_type="GNN",
        )

    def set_abar(self, abar):
        self.classifier.set_abar(abar)

    def set_SFV(self, SFV):
        self.classifier.set_SFV(SFV)

    def reset_neighgen_parameters(self):
        self.neighgen.reset_parameters()

    def get_neighgen_weights(self):
        return self.neighgen.state_dict()

    def set_neighgen_weights(self, weights):
        self.neighgen.load_state_dict(weights)

    def initialize_neighgen(self) -> None:
        self.neighgen.prepare_data(
            x=self.graph.x,
            y=self.graph.y,
            edges=self.graph.get_edges(),
            node_ids=self.graph.node_ids,
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
            selected_neighbors_ids = np.random.choice(neighbors_ids, config.num_pred)
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

    def fit_neighgen(self, epochs, inter_client_features_creators: list = []) -> None:
        self.neighgen.fit(epochs, inter_client_features_creators)

    def train_neighgen(self, inter_client_features_creators: list = []) -> None:
        self.initialize_neighgen()
        self.fit_neighgen(config.model.epoch_classifier, inter_client_features_creators)

    def initialize_locsage(self, inter_client_features_creators: list = []):
        self.train_neighgen(inter_client_features_creators)
        mend_graph = self.neighgen.create_mend_graph(self.graph)

        Client.initialize_gnn_(graph=mend_graph, classifier=self.classifier)

    def train_locsage(
        self,
        inter_client_features_creators: list = [],
        log=False,
        plot=False,
    ):
        self.initialize_locsage(inter_client_features_creators)

        return self.classifier.fit(
            epochs=config.model.epoch_classifier,
            log=log,
            plot=plot,
            type="locsage",
        )
