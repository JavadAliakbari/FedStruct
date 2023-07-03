from ast import List
from copy import deepcopy
import numpy as np

from tqdm import tqdm

from src.utils import config
from src.client import Client
from src.utils.graph import Graph
from src.classifier import Classifier
from src.structure_predictor import StructurePredictor


class Server(Client):
    def __init__(
        self,
        graph: Graph,
        num_classes,
        logger=None,
    ):
        super().__init__(
            subgraph_=graph,
            num_classes=num_classes,
            id="Server",
            logger=logger,
        )

        self.sd_predictor = StructurePredictor(
            id=self.id,
            edge_index=self.subgraph.get_edges(),
            node_ids=self.subgraph.node_ids,
        )

        self.clients: List[Client] = []
        self.num_clients = 0

        # self.subgraph.add_structural_features()

    def add_client(self, subgraph):
        client = Client(
            subgraph_=subgraph,
            id=self.num_clients,
            num_classes=self.num_classes,
            logger=self.LOGGER,
        )

        self.clients.append(client)
        self.num_clients += 1

    def train_local_classifiers(self):
        for client in self.clients:
            self.LOGGER.info(f"local classifier for client{client.id}")
            client.train_local_classifier()
            self.LOGGER.info(
                f"Client{client.id} test accuracy: {client.test_local_classifier()}"
            )

    def train_locsages(self):
        for client in self.clients:
            self.LOGGER.info(f"locsage for client{client.id}")
            client.train_locsage(bar=True, plot=True)
            self.LOGGER.info(
                f"Client{client.id} test accuracy: {client.test_local_classifier()}"
            )

    def add_weights(sum_weights, weights):
        if sum_weights is None:
            sum_weights = deepcopy(weights)
        else:
            for layer_name, layer_parameters in weights.items():
                for component_name, component_parameters in layer_parameters.items():
                    sum_weights[layer_name][component_name] += component_parameters
        return sum_weights

    def calc_mean_weights(sum_weights, count):
        for layer_name, layer_parameters in sum_weights.items():
            for component_name, component_parameters in layer_parameters.items():
                sum_weights[layer_name][component_name] = component_parameters / count

        return sum_weights

    def reset_sd_predictor_parameters(self):
        self.sd_predictor.reset_parameters()

    def get_sd_predictor_weights(self):
        return self.sd_predictor.state_dict()

    def set_sd_predictor_weights(self, weights):
        self.sd_predictor.load_state_dict(weights)

    def initialize_sd_predictor(self) -> None:
        self.sd_predictor.initialize_structural_graph(
            self.num_clients,
            self.subgraph.num_features,
            self.num_classes,
        )

    def fit_sd_predictor(self, epochs, bar=False, plot=False) -> None:
        return self.sd_predictor.fit(epochs=epochs, bar=bar, plot=plot)

    def train_sd_predictor(self) -> None:
        self.initialize_sd_predictor()
        return self.fit_sd_predictor(config.epoch_classifier, bar=True, plot=True)

    def joint_train(self, epochs=1):
        self.initialize_sd_predictor()
        plot_results = {}
        for client in self.clients:
            plot_results[f"Client{client.id}"] = []

        bar = tqdm(total=epochs)
        metrics = {}
        for epoch in range(epochs):
            res = self.sd_predictor.joint_train(self.clients, 1)
            metrics["Total Loss"] = res["Total Loss"]
            metrics["Structure Loss"] = res["Structure Loss"]

            for client in self.clients:
                result = res[f"client{client.id}"]
                result["Epoch"] = epoch
                plot_results[f"Client{client.id}"].append(result)
                metrics[f"Client{client.id}"] = result["Val Acc"]
                if epoch == epochs - 1:
                    self.LOGGER.info(f"fedsage results for client{client.id}:")
                    self.LOGGER.info(f"{result}")

            bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
            bar.set_postfix(metrics)
            bar.update()

        for client in self.clients:
            # self.LOGGER.info(
            #     f"Clinet{client.id} test accuracy: {client.test_local_classifier()}"
            # )

            Classifier.plot_results(
                plot_results[f"Client{client.id}"],
                client.id,
                type="fedsage",
            )
        # metrics = {}
        # subgraphs = []
        # for client in self.clients:
        #     subgraphs.append(client.subgraph)

        # for epoch in range(epochs):
        #     embeddibgs, structure_loss = self.sd_predictor.train(subgraphs)
        #     ind = 0
        #     for client in self.clients:
        #         node_ids = client.get_nodes()

        #         client_structure_embeddings = embeddibgs[node_ids]
        #         client_structure_loss = structure_loss[node_ids]

        #         if ind == len(self.clients) - 1:
        #             retain_graph = False
        #         else:
        #             retain_graph = True

        #         result = client.joint_train_classifier(
        #             client_structure_embeddings,
        #             client_structure_loss,
        #             retain_graph=retain_graph,
        #         )
        #         ind += 1

        #         metrics[f"client{client.id}"] = result

        # return metrics

    def train_fedSage(self, epochs=config.epoch_classifier):
        self.LOGGER.info("FedSage starts!")
        input_dimension = self.subgraph.num_features + config.structure_dim_out
        self.initialize_classifier(input_dimension=input_dimension)
        self.reset_classifier_parameters()
        self.initialize_sd_predictor()
        self.reset_sd_predictor_parameters()

        plot_results = {}
        for client in self.clients:
            client.initialize_classifier(input_dimension=input_dimension)
            client.reset_classifier_parameters()
            plot_results[f"Client{client.id}"] = []

        bar = tqdm(total=epochs)
        metrics = {}
        for epoch in range(epochs):
            weights = self.get_classifier_weights()
            sum_weights = None

            for client in self.clients:
                client.set_classifier_weights(weights)

            res = self.joint_train(config.epochs_local)
            for client in self.clients:
                new_weights = client.get_classifier_weights()
                sum_weights = Server.add_weights(sum_weights, new_weights)

                result = res[f"client{client.id}"]
                result["Epoch"] = epoch
                plot_results[f"Client{client.id}"].append(result)
                metrics[f"Client{client.id}"] = result["Val Acc"]
                if epoch == epochs - 1:
                    self.LOGGER.info(f"fedsage results for client{client.id}:")
                    self.LOGGER.info(f"{result}")

            mean_weights = Server.calc_mean_weights(sum_weights, self.num_clients)
            self.set_classifier_weights(mean_weights)
            bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
            bar.set_postfix(metrics)
            bar.update()

        for client in self.clients:
            # self.LOGGER.info(
            #     f"Clinet{client.id} test accuracy: {client.test_local_classifier()}"
            # )

            Classifier.plot_results(
                plot_results[f"Client{client.id}"],
                client.id,
                type="fedsage",
            )

        # self.LOGGER.info(f"Server test accuracy: {self.test_local_classifier()}")
