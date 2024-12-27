import logging

import numpy as np
import networkx as nx
from tqdm import tqdm

from src import *
from src.FedPub.utils import *
from src.FedPub.nets import *
from src.FedPub.fedpub_client import FedPubClient
from src.utils.graph import Graph


class FedPubServer:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.reset_clients()

    def add_client(self, subgraph):
        client: FedPubClient = FedPubClient(subgraph, self.num_clients, self.proxy)
        self.clients.append(client)
        self.num_clients += 1

    def reset_clients(self):
        self.model = MaskedGCN(
            self.graph.num_features,
            config.fedpub.n_dims,
            self.graph.num_classes,
            config.fedpub.l1,
        )
        self.model.to(device)

        self.proxy = self.get_proxy_data(self.graph.num_features)
        # self.create_workers(self.proxy)
        self.update_lists = []
        self.sim_matrices = []
        self.clients = []
        self.num_clients = 0

    def get_proxy_data(self, n_feat):
        num_graphs, num_nodes = config.fedpub.n_proxy, 100
        data = from_networkx(
            nx.random_partition_graph(
                [num_nodes] * num_graphs, p_in=0.1, p_out=0, seed=42
            )
        )
        data.x = torch.normal(mean=0, std=1, size=(num_nodes * num_graphs, n_feat))
        return data.to(device)

    def num_nodes(self):
        return self.graph.num_nodes

    def start(
        self,
        iterations=config.fedpub.epochs,
        log=True,
        plot=True,
        model_type="FedPub",
    ):
        if log:
            LOGGER.info(f"{model_type} starts!")
            bar = tqdm(total=iterations, position=0)

        coef = [client.num_nodes() / self.num_nodes() for client in self.clients]

        average_results = []
        # for client in self.clients:
        #     client.update(server_weights)

        server_weights = self.get_weights()["model"]
        almw = [server_weights for _ in self.clients]
        for curr_rnd in range(iterations):
            ##################################################
            clients_data, results = self.train_clients(curr_rnd, almw)
            average_result = sum_lod(results, coef)
            average_result["Epoch"] = curr_rnd + 1
            average_results.append(average_result)
            # LOGGER.info(f"all clients have been uploaded ({time.time()-st:.2f}s)")
            ###########################################
            almw = self.update(clients_data)
            ###########################################
            # LOGGER.info(f"[main] round {curr_rnd} done ({time.time()-st:.2f} s)")
            if log:
                bar.set_postfix(average_result)
                bar.update()

                if curr_rnd == iterations - 1:
                    self.report_results(results, "Joint Training")

        # LOGGER.info("[main] server done")
        if plot:
            title = f"Average joint Training {model_type}"
            plot_path = f"{save_path}/plots/{now}/"
            plot_metrics(average_results, title=title, save_path=plot_path)

        # if log:
        #     self.report_server_test()
        test_results = self.test_clients()
        average_result = sum_lod(test_results, coef)
        final_results = {}
        for cleint, test_result in zip(self.clients, test_results):
            final_results[f"Client{cleint.id}"] = test_result
        final_results["Average"] = average_result
        if log:
            self.report_test_results(final_results)

        return final_results

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
        sim_matrix = torch.empty(size=(n_connected, n_connected))
        for i in range(n_connected):
            for j in range(n_connected):
                lfe_i = local_functional_embeddings[i]
                lfe_j = local_functional_embeddings[j]
                a = torch.nn.functional.cosine_similarity(lfe_i, lfe_j, dim=0).item()
                sim_matrix[i, j] = a
                # b = 1 - cosine(lfe_i.cpu().numpy(), lfe_j.cpu().numpy())
                # print(f"a: {a}, b: {b}")

        if config.fedpub.agg_norm == "exp":
            sim_matrix = torch.softmax(config.fedpub.norm_scale * sim_matrix, dim=1)
        else:
            row_sums = sim_matrix.sum(axis=1)
            sim_matrix = sim_matrix / row_sums[:, torch.newaxis]

        # st = time.time()
        ratio = (torch.tensor(local_train_sizes) / sum(local_train_sizes)).tolist()
        self.set_weights(self.model, aggregate(local_weights, ratio))
        # LOGGER.info(f"global model has been updated ({time.time()-st:.2f}s)")

        # st = time.time()
        almw = []
        for client in self.clients:
            aggr_local_model_weights = aggregate(
                local_weights, sim_matrix[client.id, :]
            )
            almw.append(aggr_local_model_weights)
            # client.update(aggr_local_model_weights)

        # self.update_lists.append(updated)
        self.sim_matrices.append(sim_matrix)
        return almw
        # LOGGER.info(f"local model has been updated ({time.time()-st:.2f}s)")

    def train_clients(self, curr_rnd, almw):
        results = []
        clients_data = []

        client: FedPubClient
        for client, state_dict in zip(self.clients, almw):
            data, result = client.get_train_results(curr_rnd, state_dict)
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
                LOGGER.info(f"{client_id} {key}: {val:0.4f}")

    # def report_server_test(self):
    #     test_acc, test_loss = self.test_classifier()
    #     LOGGER.info(f"Server test: {test_acc:0.4f}")

    def test_clients(self):
        results = []
        client: FedPubClient
        for client in self.clients:
            result = client.get_test_results()
            results.append(result)

        return results

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict)

    def get_weights(self):
        return {
            "model": get_state_dict(self.model),
        }
