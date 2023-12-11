from ast import List

import torch
from tqdm import tqdm

from src.utils.utils import *
from src.client import Client
from src.utils.graph import Graph
from src.utils.config_parser import Config
from src.GNN_classifier import GNNClassifier
from src.MLP_classifier import MLPClassifier

config = Config()


class Server(Client):
    def __init__(
        self,
        graph: Graph,
        num_classes,
        classifier_type="GNN",
        save_path="./",
        logger=None,
    ):
        super().__init__(
            graph=graph,
            num_classes=num_classes,
            id="Server",
            classifier_type=classifier_type,
            save_path=save_path,
            logger=logger,
        )
        self.clients: List[Client] = []
        self.num_clients = 0

    def add_client(self, subgraph):
        client = Client(
            graph=subgraph,
            id=self.num_clients,
            num_classes=self.num_classes,
            classifier_type=self.classifier_type,
            save_path=self.save_path,
            logger=self.LOGGER,
        )

        self.clients.append(client)
        self.num_clients += 1

    def remove_clients(self):
        self.clients.clear()
        self.num_clients = 0

    def initialize_FL(
        self,
        propagate_type=config.model.propagate_type,
        structure=False,
    ) -> None:
        self.initialize(
            propagate_type=propagate_type,
            structure=structure,
        )
        client: Client
        for client in self.clients:
            client.initialize(
                propagate_type=propagate_type,
                structure=structure,
                get_structure_embeddings=self.get_structure_embeddings2,
            )

        if structure:
            self.graph.add_structural_features(
                structure_type=config.structure_model.structure_type,
                num_structural_features=config.structure_model.num_structural_features,
            )

            self.set_SFV(self.graph.structural_features)

            if propagate_type == "MP":
                abar = self.obtain_a()

                self.share_abar(abar)
                self.set_abar(abar)

                self.share_SFV()

        self.initialized = True

    def obtain_a(self):
        if config.structure_model.estimate:
            abar = estimate_a(
                self.graph.edge_index,
                self.graph.num_nodes,
                config.structure_model.mp_layers,
            )
        else:
            abar = obtain_a(
                self.graph.edge_index,
                self.graph.num_nodes,
                config.structure_model.mp_layers,
            )

        # abar_ = abar.to_dense().numpy()
        # abar1_ = abar1.to_dense().numpy()
        # e = np.mean(np.abs(abar_ - abar1_) ** 2)
        # print(e)

        return abar

    def share_abar(self, abar):
        num_nodes = self.graph.num_nodes
        row, col, val = abar.coo()

        client: Client
        for client in self.clients:
            nodes = client.get_nodes()
            node_map = client.graph.node_map

            cond = torch.isin(row, nodes)
            row_i = row[cond]
            row_i = torch.tensor(itemgetter(*np.array(row_i))(node_map))
            col_i = col[cond]
            val_i = val[cond]
            abar_i = SparseTensor(
                row=row_i,
                col=col_i,
                value=val_i,
                sparse_sizes=(len(nodes), num_nodes),
            )

            client.set_abar(abar_i)

    def share_weights(self):
        server_weights = self.state_dict()

        client: Client
        for client in self.clients:
            client.load_state_dict(server_weights)

    def report_results(self, results, framework=""):
        client: Client
        for client, result in zip(self.clients, results):
            client.report_result(result, framework)

    def get_weights(self):
        pass

    def create_SFV(self):
        pass

    def share_SFV(self):
        SFV = self.graph.structural_features

        client: Client
        for client in self.clients:
            client.set_SFV(SFV)

    def share_grads(self, grads):
        client: Client
        for client in self.clients:
            client.set_grads(grads)

        self.set_grads(grads)

    def set_train_mode(self, mode: bool = True):
        self.train(mode)

        client: Client
        for client in self.clients:
            client.train(mode)

    def train_clients(self, scale=False):
        results = []

        client: Client
        for client in self.clients:
            result = client.get_train_results(scale)
            results.append(result)

        return results

    def test_clients(self):
        results = {}

        client: Client
        for client in self.clients:
            result = client.get_test_results()
            results[f"Client{client.id}"] = result

        return results

    def report_test_results(self, test_results):
        for client_id, result in test_results.items():
            for key, val in result.items():
                self.LOGGER.info(f"{client_id} {key}: {val:0.4f}")

    def report_average_results(self, test_results, log=True):
        sum = sum_dictofdicts(test_results)
        for key, val in sum.items():
            self.LOGGER.info(f"Average {key}: {val / len(test_results):0.4f}")

    def server_test_report(self, log=True):
        test_acc = self.test_classifier()
        if log:
            self.LOGGER.info(f"Server test: {test_acc:0.4f}")

    def update_models(self):
        client: Client
        for client in self.clients:
            client.update_model()

        self.update_model()

    def reset_trainings(self):
        self.reset_model()
        client: Client
        for client in self.clients:
            client.reset_model()

    def joint_train_g(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        log=True,
        plot=True,
        structure=False,
        FL=True,
    ):
        self.LOGGER.info("joint training starts!")
        # if self.initialized:
        #     self.reset_sd_predictor_parameters()
        # else:
        #     self.initialize_sd_predictor(
        #         propagate_type=propagate_type, structure=structure
        #     )
        self.initialize_FL(propagate_type=propagate_type, structure=structure)

        if FL:
            self.share_weights()

        if log:
            bar = tqdm(total=epochs, position=0)

        average_results = []
        for epoch in range(epochs):
            self.reset_trainings()

            self.set_train_mode()
            results = self.train_clients(True & FL)
            average_result = calc_average_result(results)
            average_result["Epoch"] = epoch + 1
            average_results.append(average_result)

            if FL:
                clients_grads = get_grads(self.clients)
                grads = sum_grads(clients_grads, self.num_nodes())
                self.share_grads(grads)

            self.update_models()

            if log:
                bar.set_postfix(average_result)
                bar.update()

                if epoch == epochs - 1:
                    self.report_results(results, "Joint Training")

        if plot:
            title = f"Average joint Training G GNN"
            plot_metrics(average_results, title=title, save_path=self.save_path)

        self.server_test_report(log)
        test_results = self.test_clients()
        final_result = self.report_test_results(test_results)
        self.report_average_results(test_results, log)

        return final_result

    def joint_train_w(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        log=True,
        plot=True,
        structure=False,
        FL=True,
    ):
        self.LOGGER.info("joint training starts!")
        # if self.initialized:
        #     self.reset_sd_predictor_parameters()
        # else:
        #     self.initialize_sd_predictor(
        #         propagate_type=propagate_type, structure=structure
        #     )
        self.initialize_FL(propagate_type=propagate_type, structure=structure)

        if log:
            bar = tqdm(total=epochs, position=0)

        average_results = []
        for epoch in range(epochs):
            if FL:
                self.share_weights()
            self.reset_trainings()

            self.set_train_mode()
            results = self.train_clients()
            average_result = calc_average_result(results)
            average_result["Epoch"] = epoch + 1
            average_results.append(average_result)

            if FL:
                clients_grads = get_grads(self.clients, True)
                grads = sum_grads(clients_grads)
                self.share_grads(grads)

            self.update_models()

            if FL:
                mean_weights = sum_weights(self.clients)
                self.load_state_dict(mean_weights)

            if log:
                bar.set_postfix(average_result)
                bar.update()

                if epoch == epochs - 1:
                    self.report_results(results, "Joint Training")

        if plot:
            title = f"Average Joint Training W GNN"
            plot_metrics(average_results, title=title, save_path=self.save_path)

        self.server_test_report(log)
        test_results = self.test_clients()
        final_result = self.report_test_results(test_results)
        self.report_average_results(test_results, log)

        return final_result

    def train_locsages(self, log=True, plot=True):
        client: Client
        for client in self.clients:
            self.LOGGER.info(f"locsage for client{client.id}")
            client.train_locsage(log=log, plot=plot)
            self.LOGGER.info(
                f"Client{client.id} test accuracy: {client.test_classifier()}"
            )

    def train_fedgen(self):
        client: Client
        other_client: Client
        for client in self.clients:
            inter_client_features_creators = []
            for other_client in self.clients:
                if other_client.id != client.id:
                    inter_client_features_creators.append(
                        other_client.create_inter_features
                    )

            client.train_neighgen(inter_client_features_creators)

    def train_fedSage_plus(self, epochs=config.model.epoch_classifier):
        self.LOGGER.info("FedSage+ starts!")
        criterion = torch.nn.CrossEntropyLoss()
        self.initialize()
        self.reset_parameters()

        client: Client
        other_client: Client
        for client in self.clients:
            inter_client_features_creators = []
            for other_client in self.clients:
                if other_client.id != client.id:
                    inter_client_features_creators.append(
                        other_client.create_inter_features
                    )

            client.initialize_locsage(inter_client_features_creators)
            client.reset_parameters()

        server_results = []
        average_results = []

        bar = tqdm(total=epochs)
        for epoch in range(epochs):
            weights = self.state_dict()
            metrics = {}
            self.classifier.eval()
            if self.classifier_type == "GNN":
                (train_loss, train_acc, _, val_loss, val_acc, _) = GNNClassifier.step(
                    x=self.graph.x,
                    y=self.graph.y,
                    edge_index=self.graph.edge_index,
                    model=self.classifier.feature_model,
                    criterion=criterion,
                    train_mask=self.graph.train_mask,
                    val_mask=self.graph.val_mask,
                )
            elif self.classifier_type == "MLP":
                (train_loss, train_acc, _, val_loss, val_acc, _) = MLPClassifier.step(
                    x=self.graph.x,
                    y=self.graph.y,
                    # client.subgraph.edge_index
                    model=self.classifier.feature_model,
                    criterion=criterion,
                    train_mask=self.graph.train_mask,
                    val_mask=self.graph.val_mask,
                )

            result = {
                "Train Loss": round(train_loss.item(), 4),
                "Train Acc": round(train_acc, 4),
                "Val Loss": round(val_loss.item(), 4),
                "Val Acc": round(val_acc, 4),
                "Epoch": epoch,
            }

            metrics[f"server train acc"] = result["Train Acc"]
            metrics[f"server val acc"] = result["Val Acc"]
            server_results.append(result)
            if epoch == epochs - 1:
                self.LOGGER.info(f"fedsage+ results for client{self.id}:")
                self.LOGGER.info(f"{result}")

            sum_weights = None
            average_result = {}
            for client in self.clients:
                client.load_state_dict(weights)
                res = client.fit(config.model.epochs_local)
                new_weights = client.state_dict()
                sum_weights = add_weights(sum_weights, new_weights)

                result = res[-1]

                ratio = client.num_nodes() / self.num_nodes()
                for key, val in result.items():
                    if key not in average_result.keys():
                        average_result[key] = ratio * val
                    else:
                        average_result[key] += ratio * val

                if epoch == epochs - 1:
                    self.LOGGER.info(f"fedsage+ results for client{client.id}:")
                    self.LOGGER.info(f"{result}")

            mean_weights = calc_mean_weights(sum_weights, self.num_clients)
            self.load_state_dict(mean_weights)

            average_result["Epoch"] = epoch + 1
            average_results.append(average_result)

            metrics[f"average train acc"] = average_result["Train Acc"]
            metrics[f"average val acc"] = average_result["Val Acc"]

            bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
            bar.set_postfix(metrics)
            bar.update()

        self.LOGGER.info(f"Server test accuracy: {self.test_classifier():0.4f}")

        average_test_acc = 0
        for client in self.clients:
            test_acc = client.test_classifier()
            self.LOGGER.info(f"Clinet{client.id} test accuracy: {test_acc:0.4f}")

            average_test_acc += test_acc * client.num_nodes() / self.num_nodes()
        self.LOGGER.info(f"Average test accuracy: {average_test_acc:0.4f}")

        title = f"Server fedsage+ {self.classifier_type}"
        plot_metrics(server_results, title=title, save_path=self.save_path)

        title = f"Average fedsage+ {self.classifier_type}"
        plot_metrics(average_results, title=title, save_path=self.save_path)
