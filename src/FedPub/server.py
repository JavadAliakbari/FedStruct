import sys
import time
import numpy as np

from scipy.spatial.distance import cosine

from src.FedPub.utils import *
from src.FedPub.nets import *
from src.FedPub.federated import ServerModule
from src.FedPub.client import Client
from src.utils.graph import Graph


class Server(ServerModule):
    def __init__(self, graph: Graph, args):
        super(Server, self).__init__(args)

        self.graph = graph

        self.model = MaskedGCN(
            self.graph.num_features,
            self.args.n_dims,
            self.graph.num_classes,
            self.args.l1,
            self.args,
        )

        self.proxy = self.get_proxy_data(self.graph.num_features)
        # self.create_workers(self.proxy)
        self.update_lists = []
        self.sim_matrices = []
        self.clients = []
        self.num_clients = 0

    def add_client(self, subgraph):
        client: Client = Client(subgraph, self.args, self.num_clients, self.proxy)
        self.clients.append(client)
        self.num_clients += 1

    def get_proxy_data(self, n_feat):
        import networkx as nx

        num_graphs, num_nodes = self.args.n_proxy, 100
        data = from_networkx(
            nx.random_partition_graph(
                [num_nodes] * num_graphs, p_in=0.1, p_out=0, seed=self.args.seed
            )
        )
        data.x = torch.normal(mean=0, std=1, size=(num_nodes * num_graphs, n_feat))
        return data

    def create_workers(self, proxy):
        self.clients = []
        for client_id in range(self.args.n_clients):
            client: Client = Client(self.args, client_id, proxy)
            self.clients.append(client)

    def start(self):
        if os.path.isdir(self.args.checkpt_path) == False:
            os.makedirs(self.args.checkpt_path)
        if os.path.isdir(self.args.log_path) == False:
            os.makedirs(self.args.log_path)
        self.n_connected = round(self.args.n_clients * self.args.frac)
        for curr_rnd in range(self.args.n_rnds):
            self.curr_rnd = curr_rnd
            np.random.seed(self.args.seed + curr_rnd)
            self.selected = np.random.choice(
                self.args.n_clients, self.n_connected, replace=False
            ).tolist()
            st = time.time()
            ##################################################
            self.on_round_begin(curr_rnd)
            ##################################################

            clients_results = []
            client: Client
            for client in self.clients:
                # client.switch_state()
                # client.on_receive_message(curr_rnd)
                res = client.on_round_begin(curr_rnd)
                clients_results.append(res)
                # client.listen(curr_rnd)
            # print(f'[main] all clients updated at round {curr_rnd}')
            self.logger.print(f"all clients have been uploaded ({time.time()-st:.2f}s)")
            ###########################################
            self.on_round_complete(clients_results)
            ###########################################
            print(f"[main] round {curr_rnd} done ({time.time()-st:.2f} s)")

        print("[main] server done")

    def on_round_begin(self, curr_rnd):
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd
        # return self.get_weights()

    def on_round_complete(self, clients_results):
        self.update(clients_results)
        # self.save_state()

    def update(self, clients_results):
        st = time.time()
        local_weights = []
        local_functional_embeddings = []
        local_train_sizes = []
        client: Client
        for i in range(self.args.n_clients):
            local_weights.append(clients_results[i]["model"])
            local_functional_embeddings.append(
                clients_results[i]["functional_embedding"]
            )
            local_train_sizes.append(clients_results[i]["train_size"])

        n_connected = round(self.args.n_clients * self.args.frac)
        assert n_connected == len(local_functional_embeddings)
        sim_matrix = np.empty(shape=(n_connected, n_connected))
        for i in range(n_connected):
            for j in range(n_connected):
                lfe_i = local_functional_embeddings[i]
                lfe_j = local_functional_embeddings[j]
                sim_matrix[i, j] = 1 - cosine(lfe_i, lfe_j)

        if self.args.agg_norm == "exp":
            sim_matrix = np.exp(self.args.norm_scale * sim_matrix)

        row_sums = sim_matrix.sum(axis=1)
        sim_matrix = sim_matrix / row_sums[:, np.newaxis]

        st = time.time()
        ratio = (np.array(local_train_sizes) / np.sum(local_train_sizes)).tolist()
        self.set_weights(self.model, self.aggregate(local_weights, ratio))
        self.logger.print(f"global model has been updated ({time.time()-st:.2f}s)")

        st = time.time()
        for client in self.clients:
            aggr_local_model_weights = self.aggregate(
                local_weights, sim_matrix[client.client_id, :]
            )
            client.update(aggr_local_model_weights)

        # self.update_lists.append(updated)
        self.sim_matrices.append(sim_matrix)
        self.logger.print(f"local model has been updated ({time.time()-st:.2f}s)")

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict)

    def get_weights(self):
        return {
            "model": get_state_dict(self.model),
        }

    def save_state(self):
        torch_save(
            self.args.checkpt_path,
            "server_state.pt",
            {
                "model": get_state_dict(self.model),
                "sim_matrices": self.sim_matrices,
                "update_lists": self.update_lists,
            },
        )
