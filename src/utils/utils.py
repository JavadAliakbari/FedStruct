from ast import Dict
from collections import defaultdict
from operator import itemgetter
import os
from copy import deepcopy
from statistics import mean
import numpy as np
from scipy.sparse import coo_matrix

import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch_sparse import SparseTensor
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import degree
from tqdm import tqdm

from src.models.GNN_models import calc_accuracy, calc_f1_score

plt.rcParams["figure.figsize"] = [16, 9]
plt.rcParams["figure.dpi"] = 100  # 200 e.g. is really fine, but slower
plt.rcParams.update({"figure.max_open_warning": 0})
plt.rcParams["font.size"] = 20


def find_neighbors_(
    node_id: int,
    edge_index: torch.Tensor,
    node_map: Dict = None,
    include_node=False,
):
    if node_map is not None:
        new_node_id = node_map[node_id]
    else:
        new_node_id = node_id
    all_neighbors = np.unique(
        np.hstack(
            (
                edge_index[1, edge_index[0] == new_node_id],
                edge_index[0, edge_index[1] == new_node_id],
            )
        )
    )

    if not include_node:
        all_neighbors = np.setdiff1d(all_neighbors, new_node_id)

    if len(all_neighbors) == 0:
        return all_neighbors

    if node_map is not None:
        inv_map = {v: k for k, v in node_map.items()}
        res = itemgetter(*all_neighbors)(inv_map)
        if len(all_neighbors) == 1:
            return [res]
        else:
            return list(res)
    else:
        return all_neighbors


def plot_metrics(
    res,
    title="",
    save_path="./",
):
    dataset = pd.DataFrame.from_dict(res)
    dataset.set_index("Epoch", inplace=True)

    # save_dir = f"./plot_results/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    loss_columns = list(filter(lambda x: x.endswith("Loss"), dataset.columns))
    dataset[loss_columns].plot()
    plot_title = f"loss {title}"
    plt.title(plot_title)
    plt.savefig(f"{save_path}{plot_title}.png")

    acc_columns = list(filter(lambda x: x.endswith("Acc"), dataset.columns))
    dataset[acc_columns].plot()
    plot_title = f"accuracy {title}"
    plt.title(plot_title)
    plt.savefig(f"{save_path}{plot_title}.png")


def obtain_a(edge_index, num_nodes, num_layers):
    # num_nodes = self.num_nodes()
    eye = torch.Tensor.repeat(torch.arange(num_nodes), [2, 1])
    abar = SparseTensor(
        row=eye[0],
        col=eye[1],
        sparse_sizes=(num_nodes, num_nodes),
    )

    # edge_index = self.subgraph.edge_index
    edge_index_ = add_self_loops(edge_index)[0]

    adj = SparseTensor(
        row=edge_index_[0],
        col=edge_index_[1],
        sparse_sizes=(num_nodes, num_nodes),
    )

    node_degree = 1 / degree(edge_index_[0], num_nodes).long()
    D = SparseTensor(
        row=eye[0],
        col=eye[1],
        value=node_degree,
        sparse_sizes=(num_nodes, num_nodes),
    )

    adj_hat = D.matmul(adj)

    for _ in range(num_layers):
        abar = adj_hat.matmul(abar)  # Sparse-dense matrix multiplication

    # row, col, v = abar.coo()
    return abar


def estimate_a(edge_index, num_nodes, num_layers, num_expriments=100):
    neighbors = [
        find_neighbors_(node, edge_index, include_node=True)
        for node in range(num_nodes)
    ]
    abar = np.zeros((num_nodes, num_nodes), dtype=float)
    for _ in tqdm(range(num_expriments)):
        for node in range(num_nodes):
            chosen_node = node
            for _ in range(num_layers):
                chosen_node = np.random.choice(neighbors[chosen_node], 1)[0]
            abar[node, chosen_node] += 1

    abar /= num_expriments
    sparse_matrix = coo_matrix(abar)
    abar = SparseTensor(
        row=torch.tensor(sparse_matrix.row, dtype=torch.long),
        col=torch.tensor(sparse_matrix.col, dtype=torch.long),
        value=torch.tensor(sparse_matrix.data, dtype=torch.float32),
        sparse_sizes=(num_nodes, num_nodes),
    )
    # abar = torch.from_numpy(abar).to_sparse()

    return abar


def calc_metrics(y, y_pred, mask, criterion: None):
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    loss = criterion(y_pred[mask], y[mask])

    with torch.no_grad():
        acc = calc_accuracy(y_pred[mask].argmax(dim=1), y[mask])
        f1_score = calc_f1_score(y_pred[mask].argmax(dim=1), y[mask])

    return loss, acc, f1_score


def lod2dol(list_of_dicts):
    # convert list of dicts to dict of lists
    dict_of_lists = defaultdict(
        list,
        {key: [d[key] for d in list_of_dicts] for key in set().union(*list_of_dicts)},
    )

    return dict_of_lists


def sum_dictofdicts(dictofdicts):
    sums_per_key = {}
    # Iterate through the outer dictionary
    for outer_key, inner_dict in dictofdicts.items():
        # Iterate through the inner dictionary
        for inner_key, value in inner_dict.items():
            # Add the value to the sum for the inner key
            sums_per_key[inner_key] = sums_per_key.get(inner_key, 0) + value

    return sums_per_key


def add_weights(sum_weights, weights):
    if sum_weights is None:
        sum_weights = deepcopy(weights)
    else:
        for key, val in weights.items():
            if isinstance(val, torch.Tensor):
                sum_weights[key] += val
            else:
                add_weights(sum_weights[key], val)

    return sum_weights


def calc_mean_weights(sum_weights, count):
    for key, val in sum_weights.items():
        if isinstance(val, torch.Tensor):
            sum_weights[key] = val / count
        else:
            calc_mean_weights(sum_weights[key], count)

    return sum_weights


def sum_weights(clients):
    sum_weights = None
    for client in clients:
        client_weight = client.state_dict()
        sum_weights = add_weights(sum_weights, client_weight)

    mean_weights = calc_mean_weights(sum_weights, len(clients))

    return mean_weights


def get_grads(clients, just_SFV=False):
    clients_grads = []
    for client in clients:
        grads = client.get_grads(just_SFV)
        clients_grads.append(grads)

    return clients_grads


def sum_grads(clients_grads, num_nodes=1):
    new_grads = lod2dol(clients_grads)
    grads = {}
    for key, val in new_grads.items():
        model_grads = []
        for client_grads in zip(*val):
            try:
                model_grads.append(sum(client_grads) / num_nodes)
            except:
                model_grads.append(None)

        grads[key] = model_grads

    return grads


def calc_average_result(results):
    results_dict = lod2dol(results)

    average_result = {}
    for key, val in results_dict.items():
        average_result[key] = mean(val)

    return average_result
