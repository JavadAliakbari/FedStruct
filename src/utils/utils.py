import os
from copy import deepcopy
from statistics import mean
import time
import numpy as np
from scipy.sparse import coo_matrix

import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torch_sparse import SparseTensor
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import degree, k_hop_subgraph
from tqdm import tqdm


plt.rcParams["figure.figsize"] = [16, 9]
plt.rcParams["figure.dpi"] = 100  # 200 e.g. is really fine, but slower
plt.rcParams.update({"figure.max_open_warning": 0})
plt.rcParams["font.size"] = 20

if torch.cuda.is_available():
    dev = "cuda:0"
elif torch.backends.mps.is_available():
    dev = "mps"
else:
    dev = "cpu"
os.environ["device"] = dev


def calc_accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()


@torch.no_grad()
def calc_f1_score(pred_y, y):
    # P = pred_y[y == 1]
    # Tp = ((P == 1).sum() / len(P)).item()

    f1score = f1_score(
        pred_y.data,
        y.data,
        average="micro",
        labels=torch.unique(pred_y)
        # pred_y.data, y.data, average="weighted", labels=np.unique(pred_y)
    )
    return f1score


@torch.no_grad()
def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    out = model(data.x, data.edge_index)
    # out = out[: len(data.test_mask)]
    label = data.y[: len(data.test_mask)]
    acc = calc_accuracy(out.argmax(dim=1)[data.test_mask], label[data.test_mask])
    return acc


def find_neighbors_(
    node_id: int,
    edge_index: torch.Tensor,
    include_node=False,
    flow="undirected",  # could be "undirected", "source_to_target" or "target_to_source"
):
    if flow == "undirected":
        all_neighbors = torch.unique(
            torch.hstack(
                (
                    edge_index[1, edge_index[0] == node_id],
                    edge_index[0, edge_index[1] == node_id],
                )
            )
        )
    elif flow == "source_to_target":
        all_neighbors = torch.unique(
            edge_index[0, edge_index[1] == node_id],
        )
        all_neighbors = k_hop_subgraph(node_id, 1, edge_index, flow=flow)
    elif flow == "target_to_source":
        all_neighbors = torch.unique(
            edge_index[1, edge_index[0] == node_id],
        )

    if not include_node:
        mask = all_neighbors != node_id
        all_neighbors = all_neighbors[mask]

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
    # if dev == "mps":
    #     local_dev = "cpu"
    # else:
    #     local_dev = dev
    local_dev = "cpu"

    vals = torch.ones(num_nodes, dtype=torch.float32, device=local_dev)
    eye = torch.Tensor.repeat(torch.arange(num_nodes, device=local_dev), [2, 1])
    abar = torch.sparse_coo_tensor(
        eye,
        vals,
        (num_nodes, num_nodes),
        device=local_dev,
    )

    edge_index_ = add_self_loops(edge_index)[0].to(local_dev)

    vals = torch.ones(edge_index_.shape[1], dtype=torch.float32, device=local_dev)
    adj = torch.sparse_coo_tensor(
        edge_index_,
        vals,
        (num_nodes, num_nodes),
        device=local_dev,
    )

    node_degree = 1 / degree(edge_index_[0], num_nodes).long()
    D = torch.sparse_coo_tensor(
        eye,
        node_degree,
        (num_nodes, num_nodes),
        device=local_dev,
    )
    # print("Start...........................")
    # t1 = time.time()
    adj_hat = torch.matmul(D, adj)

    for _ in range(num_layers):
        abar = torch.matmul(adj_hat, abar)  # Sparse-dense matrix multiplication

    # abar = abar.coalesce()
    # t2 = time.time()
    # print("Finished...........................")
    # print(f"total time: {t2-t1}")

    if dev != "mps":
        abar = abar.to(dev)
    return abar


def sparse_matrix_pow(mat, power):
    # eye = torch.Tensor.repeat(torch.arange(dim), [2, 1])
    res = deepcopy(mat)
    current_power = 1

    while current_power < power:
        if current_power * 2 <= power:
            res = res.matmul(res)
            current_power *= 2
        else:
            res = res.matmul(mat)
            current_power += 1

    return res


def sparse_matrix_pow2(mat, power, dim):
    binary = list(bin(power)[2:])
    binary.reverse()
    if binary[0] == "0":
        res = torch.Tensor.repeat(torch.arange(dim), [2, 1])
        res = SparseTensor(
            row=res[0],
            col=res[1],
            sparse_sizes=(dim, dim),
        )
    else:
        res = deepcopy(mat)
    powers = deepcopy(mat)
    for val in binary[1:]:
        powers = powers.matmul(powers)
        if val == "1":
            res = res.matmul(powers)

    return res


def estimate_a(edge_index, num_nodes, num_layers, num_expriments=100):
    neighbors = [
        find_neighbors_(node, edge_index, include_node=True)
        for node in range(num_nodes)
    ]
    abar = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for _ in tqdm(range(num_expriments)):
        for node in range(num_nodes):
            chosen_node = node
            for _ in range(num_layers):
                idx = np.random.randint(neighbors[chosen_node].shape[0])
                chosen_node = neighbors[chosen_node][idx]
            abar[node, chosen_node] += 1

    abar /= num_expriments
    abar = abar.to_sparse()

    return abar


def calc_metrics(y, y_pred, mask):
    criterion = torch.nn.CrossEntropyLoss()
    y_masked = y[mask]
    y_pred_masked = y_pred[mask]

    loss = criterion(y_pred_masked, y_masked)

    with torch.no_grad():
        acc = calc_accuracy(y_pred[mask].argmax(dim=1), y[mask])
        # f1_score = calc_f1_score(y_pred[mask].argmax(dim=1), y[mask])

    return loss, acc


def lod2dol(list_of_dicts):
    if len(list_of_dicts) == 0:
        return {}
    # convert list of dicts to dict of lists
    keys = list_of_dicts[0].keys()
    dict_of_lists = {key: [d[key] for d in list_of_dicts] for key in keys}

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


def sum_grads(clients_grads, coef):
    new_grads = lod2dol(clients_grads)
    grads = {}
    for key, val in new_grads.items():
        model_grads = []
        for client_grads in zip(*val):
            try:
                weighted_client_grads = [
                    weight * tensor for weight, tensor in zip(coef, client_grads)
                ]
                model_grads.append(sum(weighted_client_grads))

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


def calc_average_result2(test_results):
    sum = sum_dictofdicts(test_results)

    average_result = {}
    for key, val in sum.items():
        average_result[key] = round(val / len(test_results), 4)

    return average_result
