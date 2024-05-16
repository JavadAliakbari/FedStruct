import os
from itertools import cycle
from collections import defaultdict

from scipy import sparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sknetwork.clustering import Louvain, Leiden, KCenters, PropagationClustering
from torch_geometric.utils import subgraph
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from sklearn.cluster import k_means
import networkx as nx

from src.utils.config_parser import Config
from src.utils.graph import Graph
from src.utils.plot_graph import plot_graph
from src.utils.utils import find_neighbors_

dev = os.environ.get("device", "cpu")
device = torch.device(dev)

path = os.environ.get("CONFIG_PATH")
config = Config(path)


def plot_communities(graph, community):
    subgraphs = create_subgraps(graph, community)
    for s in subgraphs:
        plot_graph(s)
        plt.show()


def plot_subgraphs(graph, subgraphs):
    nodes_list = []
    for subgraph in subgraphs:
        nodes = np.unique(subgraph.edge_index.numpy()[:])
        nodes_list.append(nodes)

    node_color = []
    rate = 1 / (config.subgraph.num_subgraphs - 1)
    for node in range(graph.num_nodes):
        node_color.append(0.25)
        for i in range(config.subgraph.num_subgraphs):
            if node in nodes_list[i]:
                node_color[-1] = i * rate

    # plt.figure()
    # plot_graph(subgraphs[0])
    # plt.figure()
    # plot_graph(subgraphs[1])
    # plt.figure()
    # plot_graph(subgraphs[2])
    # plt.figure()
    # plot_graph(subgraphs[3])
    # plt.figure()
    # plot_graph(subgraphs[4])
    plot_graph(graph.edge_index, node_color)
    plt.show()


def find_community(edge_index):
    adjacency = to_scipy_sparse_matrix(edge_index)
    # louvain = Leiden(resolution=1, modularity="Potts", shuffle_nodes=False)
    # louvain = PropagationClustering()

    # adjacency = sparse.csr_matrix(adjacency)
    # louvain = KCenters(n_clusters=2, directed=True)

    # louvain = Louvain(resolution=1, modularity="dugue", shuffle_nodes=True)
    louvain = Louvain()
    community = louvain.fit_predict(adjacency)

    return community


def create_community_groups(community_map, node_map=None) -> dict:
    community_groups = defaultdict(list)

    for ind, community in enumerate(community_map):
        if node_map is not None:
            node_id = node_map[ind]
        else:
            node_id = ind
        community_groups[community].append(node_id)

    return community_groups


def make_groups_smaller_than_max(community_groups, group_len_max) -> dict:
    ind = 0
    while ind < len(community_groups):
        key = list(community_groups.keys())[ind]
        if len(community_groups[key]) > group_len_max:
            l1, l2 = (
                community_groups[key][:group_len_max],
                community_groups[key][group_len_max:],
            )

            community_groups[key] = l1
            community_groups[len(community_groups)] = l2

        ind += 1

    return community_groups


def assign_nodes_to_subgraphs(community_groups, num_nodes, num_subgraphs):
    max_subgraph_nodes = num_nodes // num_subgraphs
    subgraph_node_ids = {subgraph_id: [] for subgraph_id in range(num_subgraphs)}
    subgraphs = cycle(subgraph_node_ids.keys())
    current_subgraph = next(subgraphs)

    counter = 0

    for community in community_groups.keys():
        while (
            len(subgraph_node_ids[current_subgraph]) + len(community_groups[community])
            > max_subgraph_nodes + config.subgraph.delta
            or len(subgraph_node_ids[current_subgraph]) >= max_subgraph_nodes
        ):
            current_subgraph = next(subgraphs)
            # define counter to avoid stuck in the loop forever
            counter += 1
            if counter == num_subgraphs:
                return subgraph_node_ids
        subgraph_node_ids[current_subgraph] += community_groups[community]
        counter = 0

    return subgraph_node_ids


def create_subgraps(graph: Graph, subgraph_node_ids: dict):
    subgraphs = []
    for community, subgraph_nodes in subgraph_node_ids.items():
        if not isinstance(subgraph_nodes, torch.Tensor):
            node_ids = torch.tensor(subgraph_nodes, device=device)
        else:
            node_ids = subgraph_nodes
        edges = graph.original_edge_index
        edge_mask = edges.unsqueeze(2).eq(node_ids).any(2).any(0)
        edge_index = edges[:, edge_mask]

        all_nodes = torch.unique(edge_index.flatten())
        external_nodes = all_nodes[~all_nodes.unsqueeze(1).eq(node_ids).any(1)]

        if edge_index.shape[1] != 0:
            try:
                intra_edges = subgraph(
                    node_ids,
                    edge_index=edge_index,
                )[0]
            except:
                intra_edges = subgraph(
                    node_ids, edge_index=edge_index, num_nodes=max(node_ids) + 1
                )[0]

            inter_edge_mask = edge_index.unsqueeze(2).eq(external_nodes).any(2).any(0)
            inter_edges = edge_index[:, inter_edge_mask]
        else:
            intra_edges = edge_index
            inter_edges = edge_index

        # all_edges = torch.cat((intra_edges, inter_edges), dim=0)

        # node_mask = torch.isin(graph.node_ids.to("cpu"), node_ids.to("cpu"))
        node_mask = graph.node_ids.unsqueeze(1).eq(node_ids).any(1)
        sorted_node_ids = graph.node_ids[node_mask]
        if graph.x is not None:
            x = graph.x[node_mask]
        else:
            x = None

        if graph.y is not None:
            y = graph.y[node_mask]
        else:
            y = None

        if graph.train_mask is not None:
            train_mask = graph.train_mask[node_mask.cpu()]
        else:
            train_mask = None

        if graph.test_mask is not None:
            test_mask = graph.test_mask[node_mask.cpu()]
        else:
            test_mask = None

        if graph.val_mask is not None:
            val_mask = graph.val_mask[node_mask.cpu()]
        else:
            val_mask = None

        subgraph_ = Graph(
            x=x,
            y=y,
            edge_index=intra_edges,
            node_ids=sorted_node_ids,
            external_nodes=external_nodes,
            inter_edges=inter_edges,
            train_mask=train_mask,
            test_mask=test_mask,
            val_mask=val_mask,
            num_classes=graph.num_classes,
        )
        subgraphs.append(subgraph_)

    return subgraphs


def louvain_graph_cut(graph: Graph, num_subgraphs):
    community_map = find_community(graph.edge_index)

    community_groups = create_community_groups(community_map=community_map)

    group_len_max = graph.num_nodes // num_subgraphs + config.subgraph.delta

    community_groups = make_groups_smaller_than_max(community_groups, group_len_max)

    sorted_community_groups = {
        k: v
        for k, v in sorted(
            community_groups.items(), key=lambda item: len(item[1]), reverse=True
        )
    }

    subgraph_node_ids = assign_nodes_to_subgraphs(
        sorted_community_groups, graph.num_nodes, num_subgraphs
    )

    return subgraph_node_ids


def random_assign(graph: Graph, num_subgraphs):
    subgraph_id = np.random.choice(num_subgraphs, graph.num_nodes, replace=True)
    subgraph_node_ids = {
        value: np.where(subgraph_id == value)[0] for value in range(num_subgraphs)
    }

    return subgraph_node_ids


def Kmeans_cut(graph: Graph, num_subgraphs):
    X = graph.x
    _, subgraph_id, _ = k_means(X.cpu(), num_subgraphs, n_init="auto")
    community_groups = {
        value: np.where(subgraph_id == value)[0].tolist()
        for value in range(num_subgraphs)
    }

    group_len_max = graph.num_nodes // num_subgraphs + config.subgraph.delta

    community_groups = make_groups_smaller_than_max(community_groups, group_len_max)

    sorted_community_groups = {
        k: v
        for k, v in sorted(
            community_groups.items(), key=lambda item: len(item[1]), reverse=True
        )
    }

    subgraph_node_ids = assign_nodes_to_subgraphs(
        sorted_community_groups, graph.num_nodes, num_subgraphs
    )

    return subgraph_node_ids


def Metis_cut(graph: Graph, num_subgraphs):
    import metis

    edges = graph.edge_index.T.tolist()
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(graph.num_nodes))
    nx_graph.add_edges_from(edges)
    (edgecuts, community_map) = metis.part_graph(nx_graph, num_subgraphs)
    community_groups = create_community_groups(community_map=community_map)

    sorted_community_groups = {
        k: v
        for k, v in sorted(
            community_groups.items(), key=lambda item: len(item[1]), reverse=True
        )
    }

    subgraph_node_ids = assign_nodes_to_subgraphs(
        sorted_community_groups, graph.num_nodes, num_subgraphs
    )

    return subgraph_node_ids


def create_mend_graph(subgraph: Graph, graph: Graph, val=1):
    node_ids = torch.hstack((subgraph.node_ids, subgraph.external_nodes))
    edges = torch.hstack((subgraph.original_edge_index, subgraph.inter_edges))

    node_mask = graph.node_ids.unsqueeze(1).eq(node_ids).any(1)
    sorted_node_ids = graph.node_ids[node_mask]
    subgraph_node_mask = sorted_node_ids.unsqueeze(1).eq(subgraph.node_ids).any(1)
    if graph.x is not None:
        x = graph.x[node_mask]
        x[~subgraph_node_mask] = val * x[~subgraph_node_mask]
    else:
        x = None

    if graph.y is not None:
        y = graph.y[node_mask]
        y[~subgraph_node_mask] = -1
    else:
        y = None

    if graph.train_mask is not None:
        train_mask = graph.train_mask[node_mask] & subgraph_node_mask
    else:
        train_mask = None

    if graph.test_mask is not None:
        test_mask = graph.test_mask[node_mask] & subgraph_node_mask
    else:
        test_mask = None

    if graph.val_mask is not None:
        val_mask = graph.val_mask[node_mask] & subgraph_node_mask
    else:
        val_mask = None

    mend_graph = Graph(
        x=x,
        y=y,
        edge_index=edges,
        node_ids=sorted_node_ids,
        # external_nodes=external_nodes,
        # inter_edges=inter_edges,
        train_mask=train_mask,
        test_mask=test_mask,
        val_mask=val_mask,
    )

    return mend_graph


def create_mend_graph2(subgraph: Graph, graph: Graph):
    new_node_id = graph.num_nodes
    new_edges0 = []
    new_edges1 = []
    new_x = []
    for node_id in subgraph.node_ids:
        # edge_mask = subgraph.inter_edges.unsqueeze(2).eq(node_id).any(2).any(0)

        external_nodes = find_neighbors_(node_id, subgraph.inter_edges)
        num_neighbors = external_nodes.shape[0]
        # num_neighbors = min(external_nodes.shape[0], config.fedsage.num_pred)
        # external_nodes = external_nodes[:num_neighbors]

        new_x.append(0 * graph.x[external_nodes])

        l0 = num_neighbors * [node_id.item()]
        l1 = list(range(new_node_id, new_node_id + num_neighbors))
        new_edges0 += l0 + l1
        new_edges1 += l1 + l0
        new_node_id += num_neighbors

    concatenated_x = torch.cat([subgraph.x, *new_x], dim=0)
    new_edges = np.array([new_edges0, new_edges1], dtype=int)
    new_edges = torch.tensor(new_edges, device=device)
    edges = torch.hstack((subgraph.original_edge_index, new_edges))

    max_node_id = graph.num_nodes
    new_added_nodes = concatenated_x.shape[0] - subgraph.x.shape[0]
    new_node_id = max_node_id + new_added_nodes
    new_node_ids = torch.arange(
        max_node_id, new_node_id, device=subgraph.node_ids.device
    )
    node_ids = torch.hstack((subgraph.node_ids, new_node_ids))

    train_mask = torch.hstack(
        (
            subgraph.train_mask,
            torch.zeros(new_added_nodes, dtype=torch.bool, device=dev),
        )
    )
    test_mask = torch.hstack(
        (subgraph.test_mask, torch.zeros(new_added_nodes, dtype=torch.bool, device=dev))
    )
    val_mask = torch.hstack(
        (subgraph.val_mask, torch.zeros(new_added_nodes, dtype=torch.bool, device=dev))
    )

    y_shape = list(subgraph.y.shape)
    y_shape[0] = new_added_nodes
    y = torch.hstack(
        (
            subgraph.y,
            torch.zeros(y_shape, dtype=subgraph.y.dtype, device=subgraph.y.device),
        )
    )

    mend_graph = Graph(
        x=concatenated_x,
        y=y,
        edge_index=edges,
        node_ids=node_ids,
        train_mask=train_mask,
        test_mask=test_mask,
        val_mask=val_mask,
    )

    return mend_graph


def partition_graph(graph: Graph, num_subgraphs, method="random"):
    if method == "louvain":
        subgraph_node_ids = louvain_graph_cut(graph, num_subgraphs)
    elif method == "random":
        subgraph_node_ids = random_assign(graph, num_subgraphs)
    elif method == "kmeans":
        subgraph_node_ids = Kmeans_cut(graph, num_subgraphs)
    elif method == "metis":
        subgraph_node_ids = Metis_cut(graph, num_subgraphs)

    subgraphs = create_subgraps(graph, subgraph_node_ids)

    return subgraphs
