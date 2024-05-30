from src import *
from src.utils.graph import Graph
from src.GNN.fGNN import FGNN
from src.GNN.GNN_client import GNNClient
from src.fedsage.neighgen import NeighGen


class FedSAGEClient(GNNClient):
    def __init__(self, graph: Graph, id: int = 0):
        super().__init__(graph=graph, id=id)

        self.edges = self.graph.edge_index.to("cpu")
        self.x = self.graph.x.to("cpu")

        self.neighgen = NeighGen(id=self.id, x=self.x)

    def initialize(self, **kwargs) -> None:
        mend_graph = self.neighgen.get_mend_graph()
        if mend_graph is not None:
            graph = mend_graph
        else:
            graph = self.graph

        self.classifier = FGNN(graph)
        self.classifier.create_optimizer()

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
            "Train Loss": round(train_loss, 4),
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
        epochs=config.model.iterations,
        inter_client_features_creators: list = [],
        log=True,
        plot=True,
    ):
        if log:
            LOGGER.info("Neighgen training starts!")
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
            plot_path = f"{save_path}/plots/{now}/"
            plot_metrics(results, title=title, save_path=plot_path)

        test_results = self.get_neighgen_test_results()
        if log:
            for key, val in test_results.items():
                LOGGER.info(f"Client {self.id} {key}: {val:0.4f}")

        return test_results
