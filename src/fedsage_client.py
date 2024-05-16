import os

import numpy as np

from src.GNN_client import GNNClient
from src.utils.utils import *
from src.utils.config_parser import Config
from src.utils.graph import Graph
from src.neighgen import NeighGen

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

        self.edges = self.graph.edge_index.to("cpu")
        self.x = self.graph.x.to("cpu")

        self.neighgen = NeighGen(
            id=self.id,
            num_classes=self.num_classes,
            save_path=self.save_path,
            logger=self.LOGGER,
            x=self.x,
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
        elif propagate_type == "DGCN":
            self.classifier.set_DGCN_FPM(dim_in=num_input_features)

    def initialize_neighgen(self) -> None:
        self.neighgen.prepare_data(self.graph)
        self.neighgen.set_model()

    def create_inter_features(self, node_ids):
        num_train_nodes = node_ids.shape[0]
        all_nodes = torch.randperm(self.graph.num_nodes)

        node_degrees = degree(self.edges[0], self.graph.num_nodes).long()

        inter_features = []

        node_idx = 0
        num_added_features = 0
        while num_added_features < num_train_nodes:
            if node_idx >= all_nodes.shape[0]:
                all_nodes = torch.randperm(self.graph.num_nodes)
                node_idx = 0

            node_id = all_nodes[node_idx]
            if node_degrees[node_id] == 0:
                node_idx += 1
                continue

            neighbors_ids = find_neighbors_(node_id, self.edges)
            # self.edges[0, self.edges[1] == node_id]

            if neighbors_ids.shape[0] < config.fedsage.num_pred:
                selected_neighbors_ids = neighbors_ids
            else:
                # print("iiiiiii")
                rand_idx = torch.randperm(neighbors_ids.shape[0])[
                    : config.fedsage.num_pred
                ]
                selected_neighbors_ids = neighbors_ids[rand_idx]

            inter_features.append(self.graph.x[selected_neighbors_ids])
            num_added_features += 1
            node_idx += 1

        return inter_features

    def create_inter_features2(self, node_ids):
        inter_features = []
        for node_id in node_ids:
            neighbors_ids = find_neighbors_(node_id, self.graph.inter_edges)

            if neighbors_ids.shape[0] < config.fedsage.num_pred:
                selected_neighbors_ids = neighbors_ids
            else:
                # print("iiiiiii")
                rand_idx = torch.randperm(neighbors_ids.shape[0])[
                    : config.fedsage.num_pred
                ]
                selected_neighbors_ids = neighbors_ids[rand_idx]

            mask = self.graph.node_ids.unsqueeze(1).eq(selected_neighbors_ids).any(1)

            inter_features.append(self.graph.x[mask])

        return inter_features

    def create_mend_graph(self, predict=True):
        self.neighgen.create_mend_graph(predict=predict)

    def get_neighgen_train_results(
        self, inter_client_features_creators=[], predict=True
    ):
        (
            train_loss,
            val_acc_label,
            val_acc_missing,
            # val_loss_feat,
        ) = self.neighgen.train_step(inter_client_features_creators, predict=predict)

        result = {
            "Train Loss": round(train_loss.item(), 4),
            "Val Acc": round(val_acc_label, 4),
            "Val Missing Acc": round(val_acc_missing, 4),
            # "Val Features Loss": round(val_loss_feat.item(), 4),
        }

        return result

    def get_neighgen_test_results(self):
        test_acc = self.neighgen.calc_test_accuracy()

        result = {
            "Test Acc": round(test_acc, 4),
        }

        return result

    def reset_neighgen_model(self):
        self.neighgen.reset_classifier()

    def update_neighgen_model(self):
        self.neighgen.update_model()

    def neighgen_train_mode(self, mode: bool = True):
        self.neighgen.train(mode)

    def train_neighgen_model(
        self,
        epochs=config.model.epoch_classifier,
        inter_client_features_creators: list = [],
        log=True,
        plot=True,
    ):
        self.LOGGER.info("Neighgen training starts!")

        if log:
            bar = tqdm(total=epochs, position=0)

        results = []
        for epoch in range(epochs):
            self.reset_neighgen_model()

            self.neighgen_train_mode()
            result = self.get_neighgen_train_results(inter_client_features_creators)
            result["Epoch"] = epoch + 1
            results.append(result)

            self.update_neighgen_model()

            if log:
                bar.set_postfix(result)
                bar.update()

                if epoch == epochs - 1:
                    self.report_result(result, "Local Training")

        if plot:
            title = f"client {self.id} Local Training Neighgen"
            plot_metrics(results, title=title, save_path=self.save_path)

        test_results = self.get_neighgen_test_results()
        for key, val in test_results.items():
            self.LOGGER.info(f"Client {self.id} {key}: {val:0.4f}")

        return test_results
