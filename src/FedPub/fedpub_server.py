import logging
import time

import numpy as np
import networkx as nx
from scipy.spatial.distance import cosine
from tqdm import tqdm

from src.FedPub.utils import *
from src.FedPub.nets import *
from src.FedPub.fedpub_client import FedPubClient
from src.utils.config_parser import Config
from src.utils.graph import Graph
from src.utils.utils import calc_average_result, calc_average_result2, plot_metrics

path = os.environ.get("CONFIG_PATH")
config = Config(path)
now = os.environ.get("now", 0)


class FedPubServer:
    def __init__(
        self,
        graph: Graph,
        save_path="./",
        logger=None,
    ):
        self.graph = graph
        self.save_path = save_path
        self.LOGGER = logger or logging

        self.model = MaskedGCN(
            self.graph.num_features,
            config.fedpub.n_dims,
            self.graph.num_classes,
            config.fedpub.l1,
        )

        self.proxy = self.get_proxy_data(self.graph.num_features)
        # self.create_workers(self.proxy)
        self.update_lists = []
        self.sim_matrices = []
        self.clients = []
        self.num_clients = 0

    def add_client(self, subgraph):
        client: FedPubClient = FedPubClient(
            subgraph,
            self.num_clients,
            self.proxy,
            save_path=self.save_path,
            logger=self.LOGGER,
        )
        self.clients.append(client)
        self.num_clients += 1

    def remove_clients(self):
        self.clients.clear()
        self.num_clients = 0

    def get_proxy_data(self, n_feat):

        num_graphs, num_nodes = config.fedpub.n_proxy, 100
        data = from_networkx(
            nx.random_partition_graph(
                [num_nodes] * num_graphs, p_in=0.1, p_out=0, seed=42
            )
        )
        data.x = torch.normal(mean=0, std=1, size=(num_nodes * num_graphs, n_feat))
        return data

    def create_workers(self, proxy):
        self.clients = []
        for client_id in range(config.subgraph.num_subgraphs):
            client: FedPubClient = FedPubClient(client_id, proxy)
            self.clients.append(client)

    def start(
        self,
        iterations=config.fedpub.epochs,
        log=True,
        plot=True,
        model_type="FedPub",
    ):
        self.LOGGER.info(f"{model_type} starts!")

        if log:
            bar = tqdm(total=iterations, position=0)

        average_results = []
        for curr_rnd in range(iterations):
            ##################################################
            clients_data, results = self.train_clients(curr_rnd)
            average_result = calc_average_result(results)
            average_result["Epoch"] = curr_rnd + 1
            average_results.append(average_result)
            # self.LOGGER.info(f"all clients have been uploaded ({time.time()-st:.2f}s)")
            ###########################################
            self.update(clients_data)
            ###########################################
            # self.LOGGER.info(f"[main] round {curr_rnd} done ({time.time()-st:.2f} s)")
            if log:
                bar.set_postfix(average_result)
                bar.update()

                if curr_rnd == iterations - 1:
                    self.report_results(results, "Joint Training")

        # self.LOGGER.info("[main] server done")
        if plot:
            title = f"Average joint Training {model_type}"
            plot_path = f"{self.save_path}/plots/{now}/"
            plot_metrics(average_results, title=title, save_path=plot_path)

        # if log:
        #     self.report_server_test()
        test_results = self.test_clients()
        average_result = calc_average_result2(test_results)
        test_results["Average"] = average_result
        if log:
            self.report_test_results(test_results)

        return test_results

    def update(self, clients_data):
        # st = time.time()
        local_weights = []
        local_functional_embeddings = []
        local_train_sizes = []
        client: FedPubClient
        for i in range(self.num_clients):
            local_weights.append(clients_data[i]["model"])
            local_functional_embeddings.append(clients_data[i]["functional_embedding"])
            local_train_sizes.append(clients_data[i]["train_size"])

        n_connected = round(self.num_clients * config.fedpub.frac)
        assert n_connected == len(local_functional_embeddings)
        sim_matrix = np.empty(shape=(n_connected, n_connected))
        for i in range(n_connected):
            for j in range(n_connected):
                lfe_i = local_functional_embeddings[i]
                lfe_j = local_functional_embeddings[j]
                sim_matrix[i, j] = 1 - cosine(lfe_i, lfe_j)

        if config.fedpub.agg_norm == "exp":
            sim_matrix = np.exp(config.fedpub.norm_scale * sim_matrix)

        row_sums = sim_matrix.sum(axis=1)
        sim_matrix = sim_matrix / row_sums[:, np.newaxis]

        # st = time.time()
        ratio = (np.array(local_train_sizes) / np.sum(local_train_sizes)).tolist()
        self.set_weights(self.model, aggregate(local_weights, ratio))
        # self.LOGGER.info(f"global model has been updated ({time.time()-st:.2f}s)")

        # st = time.time()
        for client in self.clients:
            aggr_local_model_weights = aggregate(
                local_weights, sim_matrix[client.id, :]
            )
            client.update(aggr_local_model_weights)

        # self.update_lists.append(updated)
        self.sim_matrices.append(sim_matrix)
        # self.LOGGER.info(f"local model has been updated ({time.time()-st:.2f}s)")

    def train_clients(self, curr_rnd):
        results = []
        clients_data = []

        client: FedPubClient
        for client in self.clients:
            data, result = client.get_train_results(curr_rnd)
            clients_data.append(data)
            results.append(result)

        return clients_data, results

    def report_results(self, results, framework=""):
        client: FedPubClient
        for client, result in zip(self.clients, results):
            client.report_result(result, framework)

    def report_test_results(self, test_results):
        for client_id, result in test_results.items():
            for key, val in result.items():
                self.LOGGER.info(f"{client_id} {key}: {val:0.4f}")

    # def report_server_test(self):
    #     test_acc, test_loss = self.test_classifier()
    #     self.LOGGER.info(f"Server test: {test_acc:0.4f}")

    def test_clients(self):
        results = {}

        client: FedPubClient
        for client in self.clients:
            result = client.get_test_results()
            results[f"Client{client.id}"] = result

        return results

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict)

    def get_weights(self):
        return {
            "model": get_state_dict(self.model),
        }
