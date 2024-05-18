import os
from copy import deepcopy
from statistics import mean
import numpy as np

import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# from torch_sparse import SparseTensor
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import degree
from tqdm import tqdm
from sknetwork.clustering import Louvain
from torch_geometric.utils.convert import to_scipy_sparse_matrix

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


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


@torch.no_grad()
def calc_accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()


def calc_accuracy2(pred_y, y):
    """Calculate accuracy."""
    return sum(pred_y == y) / len(y)


@torch.no_grad()
def calc_f1_score(pred_y, y):
    # P = pred_y[y == 1]
    # Tp = ((P == 1).sum() / len(P)).item()

    f1score = f1_score(
        pred_y.data,
        y.data,
        average="micro",
        labels=torch.unique(pred_y),
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
        edge_mask = edge_index.unsqueeze(2).eq(node_id).any(2).any(0)
        edges = edge_index[:, edge_mask]
    elif flow == "source_to_target":
        edge_mask = edge_index[0].unsqueeze(1).eq(node_id).any(1)
        edges = edge_index[0, edge_mask]
    elif flow == "target_to_source":
        edge_mask = edge_index[1].unsqueeze(1).eq(node_id).any(1)
        edges = edge_index[1, edge_mask]

    all_neighbors = torch.unique(edges)
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

    os.makedirs(save_path, exist_ok=True)

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


def sparse_eye(n, vals=None, dev="cpu"):
    if vals is None:
        vals = torch.ones(n, dtype=torch.float32, device=dev)
    eye = torch.Tensor.repeat(torch.arange(n, device=dev), [2, 1])
    sparse_matrix = torch.sparse_coo_tensor(eye, vals, (n, n), device=dev)

    return sparse_matrix


def obtain_a(edge_index, num_nodes, num_layers, pruning=False):
    # if dev == "mps":
    #     local_dev = "cpu"
    # else:
    #     local_dev = dev
    local_dev = "cpu"

    abar = sparse_eye(num_nodes, dev=local_dev)

    edge_index_ = add_self_loops(edge_index)[0].to(local_dev)

    vals = torch.ones(edge_index_.shape[1], dtype=torch.float32, device=local_dev)
    adj = torch.sparse_coo_tensor(
        edge_index_,
        vals,
        (num_nodes, num_nodes),
        device=local_dev,
    )

    node_degree = 1 / degree(edge_index_[0], num_nodes).long()
    D = sparse_eye(num_nodes, node_degree, local_dev)
    adj_hat = torch.matmul(D, adj)

    # abar = sparse_matrix_pow2(adj_hat, num_layers, num_nodes)

    th = 30
    for i in range(num_layers):
        abar = torch.matmul(adj_hat, abar)  # Sparse-dense matrix multiplication
        if pruning:
            abar = prune(abar, th)
        # print(f"{i}: {abar.values().shape[0]/(num_nodes*num_nodes)}")

    if dev != "mps":
        abar = abar.to(dev)
    return abar


def prune(abar, degree):
    num_nodes = abar.shape[0]
    num_vals = num_nodes * degree
    vals = abar.values()
    if num_vals >= vals.shape[0]:
        return abar
    sorted_vals_idx = torch.argsort(vals, descending=True)
    chosen_vals_idx = sorted_vals_idx[:num_vals]
    # chosen_vals_idx = np.random.choice(vals.shape[0], num_vals, replace=False)

    # mask = abar.values() > th
    idx = abar.indices()[:, chosen_vals_idx]
    val = abar.values()[chosen_vals_idx]
    # val = torch.masked_select(abar.values(), mask)

    abar = torch.sparse_coo_tensor(idx, val, abar.shape, device=abar.device)
    abar = abar.coalesce()

    return abar


def plot_abar(abar, edge_index):
    dense_abar = abar.to_dense().numpy()
    dense_abar = np.power(dense_abar, 0.25)

    adjacency = to_scipy_sparse_matrix(edge_index)
    louvain = Louvain()
    community_map = louvain.fit_predict(adjacency)
    community_based_node_order = np.argsort(community_map)
    dense_abar = dense_abar[:, community_based_node_order]
    dense_abar = dense_abar[community_based_node_order, :]

    plt.imshow(dense_abar, cmap="gray", interpolation="nearest")
    plt.imsave("./models/CiteSeer_True.png", dense_abar)
    plt.show()


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

    acc = calc_accuracy(y_pred_masked.argmax(dim=1), y_masked)
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


def log_config(_LOGGER, config):
    _LOGGER.info(f"dataset name: {config.dataset.dataset_name}")
    _LOGGER.info(f"num subgraphs: {config.subgraph.num_subgraphs}")
    _LOGGER.info(f"partitioning method: {config.subgraph.partitioning}")
    _LOGGER.info(f"num Epochs: {config.model.iterations}")
    _LOGGER.info(f"batch: {config.model.batch}")
    _LOGGER.info(f"batch size: {config.model.batch_size}")
    _LOGGER.info(f"learning rate: {config.model.lr}")
    _LOGGER.info(f"weight decay: {config.model.weight_decay}")
    _LOGGER.info(f"dropout: {config.model.dropout}")
    _LOGGER.info(f"gnn layer type: {config.model.gnn_layer_type}")
    _LOGGER.info(f"propagate type: {config.model.propagate_type}")
    _LOGGER.info(f"gnn layer sizes: {config.feature_model.gnn_layer_sizes}")
    _LOGGER.info(f"DGCN_layer_sizes: {config.feature_model.DGCN_layer_sizes}")
    _LOGGER.info(f"mlp layer sizes: {config.feature_model.mlp_layer_sizes}")
    _LOGGER.info(f"structure DGCN layers: {config.structure_model.DGCN_layers}")
    _LOGGER.info(f"feature DGCN layers: {config.feature_model.DGCN_layers}")
    if config.model.propagate_type == "GNN":
        _LOGGER.info(
            f"structure layers size: {config.structure_model.GNN_structure_layers_sizes}"
        )
    else:
        _LOGGER.info(
            f"structure layers size: {config.structure_model.DGCN_structure_layers_sizes}"
        )
    _LOGGER.info(f"structure type: {config.structure_model.structure_type}")
    _LOGGER.info(
        f"num structural features: {config.structure_model.num_structural_features}"
    )
    _LOGGER.info(
        f"Train-Test ratio: [{config.subgraph.train_ratio}, {config.subgraph.test_ratio}]"
    )
