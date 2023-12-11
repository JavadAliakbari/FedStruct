import logging

import numpy as np
from tqdm import tqdm

from src.utils.utils import *
from src.utils.config_parser import Config
from src.utils.graph import Data_, Graph
from src.GNN_classifier import GNNClassifier
from src.MLP_classifier import MLPClassifier
from src.neighgen import NeighGen
from src.models.feature_loss import greedy_loss

config = Config()
config.num_pred = 5


class Client:
    def __init__(
        self,
        graph: Graph,
        num_classes,
        id: int = 0,
        classifier_type="GNN",
        save_path="./",
        logger=None,
    ):
        self.id = id
        self.graph = graph
        self.num_classes = num_classes
        self.classifier_type = classifier_type
        self.save_path = save_path

        self.LOGGER = logger or logging

        self.LOGGER.info(f"Client{self.id} statistics:")
        self.LOGGER.info(f"Number of nodes: {self.graph.num_nodes}")
        self.LOGGER.info(f"Number of edges: {self.graph.num_edges}")
        self.LOGGER.info(f"Number of features: {self.graph.num_features}")

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
        return self.graph.node_ids

    def num_nodes(self) -> int:
        return len(self.graph.node_ids)

    def parameters(self):
        return self.classifier.parameters()

    def zero_grad(self):
        self.classifier.zero_grad()

    def get_structure_embeddings2(self, node_ids):
        return self.classifier.get_structure_embeddings2(node_ids)

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
        graph: Graph,
        classifier: GNNClassifier,
        propagate_type=config.model.propagate_type,
        num_input_features=None,
        structure=False,
        get_structure_embeddings=None,
    ) -> None:
        classifier.prepare_data(
            graph=graph,
            batch_size=config.model.batch_size,
            num_neighbors=config.model.num_samples,
        )

        if propagate_type == "GNN":
            classifier.set_GNN_FPM(dim_in=num_input_features)
        elif propagate_type == "MP":
            classifier.set_DGCN_FPM(dim_in=num_input_features)

        if structure:
            if config.structure_model.structure_type != "GDV":
                dim_in = config.structure_model.num_structural_features
            else:
                dim_in = 73

            if propagate_type == "GNN":
                classifier.set_GNN_SPM(
                    dim_in=dim_in,
                    get_structure_embeddings=get_structure_embeddings,
                )
            elif propagate_type == "MP":
                classifier.set_DGCN_SPM(dim_in=dim_in)

    def initialize_mlp_(
        data: Data_,
        classifier: MLPClassifier,
        dim_in=None,
    ):
        classifier.prepare_data(
            data=data,
            batch_size=config.model.batch_size,
        )
        classifier.set_classifiers(dim_in=dim_in)

    def initialize(
        self,
        propagate_type=config.model.propagate_type,
        input_dimension=None,
        structure=False,
        get_structure_embeddings=None,
    ) -> None:
        if self.classifier_type == "GNN":
            Client.initialize_gnn_(
                self.graph,
                self.classifier,
                # self.num_classes,
                propagate_type,
                input_dimension,
                structure,
                get_structure_embeddings,
            )
        elif self.classifier_type == "MLP":
            Client.initialize_mlp_(
                self.graph,
                self.classifier,
                input_dimension,
            )

    def test_classifier(self, metric=config.model.metric):
        return self.classifier.calc_test_accuracy(metric)

    def get_train_results(self, scale=False):
        (
            train_loss,
            train_acc,
            train_f1_score,
            val_loss,
            val_acc,
            val_f1_score,
        ) = self.train_step(scale)

        result = {
            "Train Loss": round(train_loss.item(), 4),
            "Train Acc": round(train_acc, 4),
            "Val Loss": round(val_loss.item(), 4),
            "Val Acc": round(val_acc, 4),
        }

        return result

    def get_test_results(self):
        test_acc = self.test_classifier()

        result = {
            "Test Acc": round(test_acc, 4),
        }

        return result

    def report_result(self, result, framework=""):
        self.LOGGER.info(f"{framework} results for client{self.id}:")
        self.LOGGER.info(f"{result}")

    def train_local_model(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        log=True,
        plot=True,
        structure=False,
    ):
        self.LOGGER.info("local training starts!")
        self.initialize(propagate_type=propagate_type, structure=structure)

        if log:
            bar = tqdm(total=epochs, position=0)

        results = []
        for epoch in range(epochs):
            self.reset_model()

            self.train()
            result = self.get_train_results()
            result["Epoch"] = epoch + 1
            results.append(result)

            self.update_model()

            if log:
                bar.set_postfix(result)
                bar.update()

                if epoch == epochs - 1:
                    self.report_result(result, "Local Training")

        if plot:
            title = f"client {self.id} Local Training"
            plot_metrics(results, title=title, save_path=self.save_path)

        test_results = self.get_test_results()
        for key, val in test_results.items():
            self.LOGGER.info(f"{self.id} {key}: {val:0.4f}")

        return test_results

    def set_abar(self, abar):
        self.classifier.set_abar(abar)

    def set_SFV(self, SFV):
        self.classifier.set_SFV(SFV)

    def get_grads(self, just_SFV=False):
        return self.classifier.get_model_grads(just_SFV)

    def set_grads(self, grads):
        self.classifier.set_model_grads(grads)

    def update_model(self):
        self.classifier.update_model()

    def reset_model(self):
        self.classifier.reset_client()

    def train_step(self, scale=False):
        return self.classifier.train_step(scale)

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
