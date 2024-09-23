import os
import random
import itertools
from ast import List
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import networkx as nx
from torch_sparse import SparseTensor
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torch_geometric.utils import degree, add_self_loops
from dotenv import load_dotenv

from src.utils.config_parser import Config
from src.utils.logger import getLOGGER

load_dotenv()
config_path = os.environ.get("CONFIGPATH")
config = Config(config_path)

plt.rcParams["figure.figsize"] = [16, 9]
plt.rcParams["figure.dpi"] = 100  # 200 e.g. is really fine, but slower
plt.rcParams.update({"figure.max_open_warning": 0})
plt.rcParams["font.size"] = 20

if torch.cuda.is_available():
    dev = "cuda:0"
# elif torch.backends.mps.is_available():
#     dev = "mps"
else:
    dev = "cpu"
device = torch.device(dev)

if dev == "mps":
    local_dev = "cpu"
else:
    local_dev = dev


seed = 45
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

now = datetime.now().strftime("%Y%m%d_%H%M%S")

result_path = os.environ.get("RESULTPATH")
save_path = (
    f"{result_path}"
    f"{config.dataset.dataset_name}/"
    f"{config.structure_model.structure_type}/"
    f"{config.subgraph.partitioning}/"
    f"{config.model.smodel_type}/"
    f"{config.subgraph.num_subgraphs}/all/"
)

os.makedirs(save_path, exist_ok=True)
LOGGER = getLOGGER(
    name=f"{now}_{config.dataset.dataset_name}",
    log_on_file=True,
    save_path=save_path,
)


@torch.no_grad()
def calc_accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()


@torch.no_grad()
def calc_f1_score(pred_y, y):
    f1score = f1_score(
        pred_y.detach().cpu().numpy(),
        y.detach().cpu().numpy(),
        average="micro",
        labels=torch.unique(pred_y),
    )
    return f1score


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


def obtain_a(edge_index, num_nodes, num_layers, estimate=False, pruning=False):
    if estimate:
        abar = estimate_a(edge_index, num_nodes, num_layers)
    else:
        abar = calc_a(edge_index, num_nodes, num_layers, pruning)

    return abar


def sparse_eye(n, vals=None, dev="cpu"):
    if vals is None:
        vals = torch.ones(n, dtype=torch.float32, device=dev)
    eye = torch.Tensor.repeat(torch.arange(n, device=dev), [2, 1])
    sparse_matrix = torch.sparse_coo_tensor(eye, vals, (n, n), device=dev)

    return sparse_matrix


def create_rw(edge_index, num_nodes, num_layers):
    local_dev = "cpu"

    abar = sparse_eye(num_nodes, dev=local_dev)

    vals = torch.ones(edge_index.shape[1], dtype=torch.float32, device=local_dev)
    adj = torch.sparse_coo_tensor(
        edge_index,
        vals,
        (num_nodes, num_nodes),
        device=local_dev,
    )

    node_degree = 1 / degree(edge_index[0], num_nodes).long()
    D = sparse_eye(num_nodes, node_degree, local_dev)
    adj_hat = torch.matmul(adj, D)

    SE = []
    for i in range(num_layers):
        abar = torch.matmul(abar, adj_hat)  # Sparse-dense matrix multiplication
        SE.append(torch.diag(abar.to_dense()))

    SE_rw = torch.stack(SE, dim=-1)

    # if dev != "mps":
    SE_rw = SE_rw.to(dev)
    return SE_rw


def calc_a(edge_index, num_nodes, num_layers, pruning=False):
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

    th = config.subgraph.pruning_th
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


def split_abar(abar: SparseTensor, nodes):
    num_nodes = abar.size()[0]
    # nodes = self.get_nodes().to(local_dev)
    num_nodes_i = len(nodes)
    indices = torch.arange(num_nodes_i, dtype=torch.long, device=local_dev)
    vals = torch.ones(num_nodes_i, dtype=torch.float32, device=local_dev)
    P = torch.sparse_coo_tensor(
        torch.vstack([indices, nodes]),
        vals,
        (num_nodes_i, num_nodes),
        device=local_dev,
    )
    abar_i = torch.matmul(P, abar)
    if dev != "cuda:0":
        abar_i = abar_i.to_dense().to(dev)
    return abar_i


def plot_abar(abar, edge_index):
    dense_abar = abar.to_dense().numpy()
    dense_abar = np.power(dense_abar, 0.25)

    G = nx.Graph(edge_index.T.tolist())
    community = nx.community.louvain_communities(G)

    sorted_community_groups = sorted(
        community, key=lambda item: len(item), reverse=True
    )
    community_based_node_order = itertools.chain.from_iterable(sorted_community_groups)

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


def calc_metrics(y, y_pred, mask, loss_function="cross_entropy"):
    y_masked = y[mask]
    y_pred_masked = y_pred[mask]

    if loss_function == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_function == "log_likelihood":
        criterion = torch.nn.NLLLoss()
    elif loss_function == "BCELoss":
        criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(y_pred_masked, y_masked)

    acc = calc_accuracy(y_pred_masked.argmax(dim=1), y_masked)
    # f1_score_ = calc_f1_score(y_pred_masked.argmax(dim=1), y_masked)

    return loss, acc


def lod2dol(list_of_dicts):
    # convert list of dicts to dict of lists
    if len(list_of_dicts) == 0:
        return {}
    keys = list_of_dicts[0].keys()
    dict_of_lists = {key: [d[key] for d in list_of_dicts] for key in keys}

    return dict_of_lists


def lol2lol(list_of_lists):
    # convert list of lists to list of lists
    if len(list_of_lists) == 0:
        return []
    keys = range(len(list_of_lists[0]))
    res = [[d[key] for d in list_of_lists] for key in keys]

    return res


def get_grads(clients, just_SFV=False):
    clients_grads = []
    for client in clients:
        grads = client.get_grads(just_SFV)
        clients_grads.append(grads)

    return clients_grads


def state_dict(clients):
    clients_weights = []
    for client in clients:
        grads = client.state_dict()
        clients_weights.append(grads)

    return clients_weights


def sum_lod(x: List, coef=None):
    if len(x) == 0:
        return x
    if coef is None:
        coef = len(x) * [1 / len(x)]
    if isinstance(x[0], dict):
        z = lod2dol(x)
        for key, val in z.items():
            z[key] = sum_lod(val, coef)
        return z
    elif isinstance(x[0], list):
        z = lol2lol(x)
        for key, val in enumerate(z):
            z[key] = sum_lod(val, coef)
        return z
    else:
        return sum([weight * val for weight, val in zip(coef, x)])
