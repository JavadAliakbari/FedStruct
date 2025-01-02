from itertools import chain
from collections import defaultdict

import torch
import numpy as np
import networkx as nx
from sklearn.cluster import k_means
from torch_geometric.utils import subgraph

from src import *
from src.utils.graph import Graph
from src.FedGCN.utils import label_dirichlet_partition, get_in_comm_indexes


def find_community(edge_index, num_nodes):
    G = nx.Graph(edge_index.T.tolist())
    community = nx.community.louvain_communities(G)
    community_noeds = torch.tensor(list(chain.from_iterable(community)))

    node_ids = torch.arange(num_nodes)
    node_mask = node_ids.unsqueeze(1).eq(community_noeds).any(1)
    isolated_nodes = node_ids[~node_mask]
    community.append(isolated_nodes)

    community = {ind: list(c) for ind, c in enumerate(community)}

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
        if len(community_groups[ind]) > group_len_max:
            l1, l2 = (
                community_groups[ind][:group_len_max],
                community_groups[ind][group_len_max:],
            )

            community_groups[ind] = l1
            community_groups[len(community_groups)] = l2

        ind += 1

    return community_groups


def assign_nodes_to_subgraphs(community_groups, num_nodes, num_subgraphs):
    max_subgraph_nodes = num_nodes // num_subgraphs
    subgraph_node_ids = {subgraph_id: [] for subgraph_id in range(num_subgraphs)}
    # subgraphs = cycle(subgraph_node_ids.keys())
    current_ind = 0

    counter = 0

    for community in community_groups.keys():
        while (
            len(subgraph_node_ids[current_ind]) + len(community_groups[community])
            > max_subgraph_nodes + config.subgraph.delta
            or len(subgraph_node_ids[current_ind]) >= max_subgraph_nodes
        ):
            # current_subgraph = next(subgraphs)
            current_ind += 1
            if current_ind == num_subgraphs:
                current_ind = 0
            # define counter to avoid stuck in the loop forever
            counter += 1
            if counter == num_subgraphs:
                current_ind = np.argmin([len(s) for s in subgraph_node_ids.values()])
                break
                # subgraph_node_ids[ind] += community_groups[community]
                # current_ind += 1
                # if current_ind == num_subgraphs:
                #     current_ind = 0
                # current_subgraph = next(subgraphs)
                # return subgraph_node_ids
        subgraph_node_ids[current_ind] += community_groups[community]
        counter = 0

    assert sum([len(s) for s in subgraph_node_ids.values()]) == num_nodes

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
            inter_edge_mask = edge_index.unsqueeze(2).eq(external_nodes).any(2).any(0)
            inter_edges = edge_index[:, inter_edge_mask]
            intra_edges = edge_index[:, ~inter_edge_mask]
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


def louvain_cut(edge_index, num_nodes, num_subgraphs):
    community_groups = find_community(edge_index, num_nodes)

    group_len_max = num_nodes // num_subgraphs + config.subgraph.delta

    community_groups = make_groups_smaller_than_max(community_groups, group_len_max)

    sorted_community_groups = {
        k: v
        for k, v in sorted(
            community_groups.items(), key=lambda item: len(item[1]), reverse=True
        )
    }

    subgraph_node_ids = assign_nodes_to_subgraphs(
        sorted_community_groups, num_nodes, num_subgraphs
    )

    return subgraph_node_ids


def random_assign(num_nodes, num_subgraphs):
    subgraph_id = np.random.choice(num_subgraphs, num_nodes, replace=True)
    subgraph_node_ids = {
        value: torch.tensor(
            np.where(subgraph_id == value)[0], dtype=torch.int64, device=dev
        )
        for value in range(num_subgraphs)
    }

    return subgraph_node_ids


def kmeans_cut(X, num_subgraphs):
    num_nodes = X.shape[0]
    _, subgraph_id, _ = k_means(X.cpu(), num_subgraphs, n_init="auto")
    community_groups = create_community_groups(subgraph_id)

    group_len_max = num_nodes // num_subgraphs + config.subgraph.delta

    community_groups = make_groups_smaller_than_max(community_groups, group_len_max)

    sorted_community_groups = {
        k: v
        for k, v in sorted(
            community_groups.items(), key=lambda item: len(item[1]), reverse=True
        )
    }

    subgraph_node_ids = assign_nodes_to_subgraphs(
        sorted_community_groups, num_nodes, num_subgraphs
    )

    return subgraph_node_ids


def metis_cut(edge_index, num_nodes, num_subgraphs):
    import metis

    edges = edge_index.T.tolist()
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(num_nodes))
    nx_graph.add_edges_from(edges)
    (edgecuts, community_map) = metis.part_graph(nx_graph, num_subgraphs)
    community_groups = create_community_groups(community_map=community_map)

    return community_groups


def drichlet_cut(labels, num_nodes, num_subgraphs, num_classes):
    subgraph_node_ids = label_dirichlet_partition(
        labels.cpu().numpy(),
        num_nodes,
        num_classes,
        num_subgraphs,
        beta=config.fedgcn.iid_beta,
    )
    subgraph_node_ids = [torch.tensor(node_ids) for node_ids in subgraph_node_ids]
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
        num_classes=graph.num_classes,
    )

    return mend_graph


def create_comm_indexes(graph: Graph, subgraph_node_ids: Graph, num_hops=2):
    # edge_index, subgraph_node_ids, train_mask, test_mask):
    train_mask = graph.train_mask
    test_mask = graph.test_mask
    edge_index = graph.edge_index
    idx = torch.arange(train_mask.shape[0])
    idx_train = graph.node_ids[train_mask]
    idx_test = graph.node_ids[test_mask]
    num_subgraphs = len(subgraph_node_ids)
    (
        communicate_indexes,
        in_com_train_data_indexes,
        in_com_test_data_indexes,
        edge_indexes_clients,
    ) = get_in_comm_indexes(
        edge_index,
        subgraph_node_ids,
        num_subgraphs,
        num_hops,
        idx_train,
        idx_test,
    )

    subgraphs = []
    for i in range(len(communicate_indexes)):
        node_mask = graph.node_ids.unsqueeze(1).eq(communicate_indexes[i]).any(1)
        x = graph.x[node_mask]
        y = graph.y[node_mask]
        subgraph_train_mask = (
            communicate_indexes[i].unsqueeze(1).eq(in_com_train_data_indexes[i]).any(1)
        )
        subgraph_test_mask = (
            communicate_indexes[i].unsqueeze(1).eq(in_com_test_data_indexes[i]).any(1)
        )
        subgraph_val_mask = ~(subgraph_train_mask | subgraph_test_mask)

        subgraph = Graph(
            x=x,
            y=y,
            edge_index=edge_indexes_clients[i],
            # node_ids=communicate_indexes[i],
            # external_nodes=external_nodes,
            # inter_edges=inter_edges,
            train_mask=subgraph_train_mask,
            test_mask=subgraph_test_mask,
            val_mask=subgraph_val_mask,
            num_classes=graph.num_classes,
        )

        subgraphs.append(subgraph)

    return subgraphs


def partition_graph(graph: Graph, num_subgraphs, method="random"):
    if method == "louvain":
        subgraph_node_ids = louvain_cut(
            graph.edge_index, graph.num_nodes, num_subgraphs
        )
    elif method == "random":
        subgraph_node_ids = random_assign(graph.num_nodes, num_subgraphs)
    elif method == "kmeans":
        subgraph_node_ids = kmeans_cut(graph.x, num_subgraphs)
    elif method == "metis":
        subgraph_node_ids = metis_cut(graph.edge_index, graph.num_nodes, num_subgraphs)

    subgraphs = create_subgraps(graph, subgraph_node_ids)

    return subgraphs


def fedGCN_partitioning(
    graph: Graph, num_subgraphs, method="drichlet", num_hops=config.fedgcn.num_hops
):
    if method == "louvain":
        subgraph_node_ids = louvain_cut(
            graph.edge_index, graph.num_nodes, num_subgraphs
        )
        subgraph_node_ids = {
            key: torch.tensor(node_ids, dtype=torch.int64, device=dev)
            for key, node_ids in subgraph_node_ids.items()
        }
    elif method == "random":
        subgraph_node_ids = random_assign(graph.num_nodes, num_subgraphs)
    elif method == "drichlet":
        subgraph_node_ids = drichlet_cut(
            graph.y, graph.num_nodes, num_subgraphs, graph.num_classes
        )
    elif method == "kmeans":
        subgraph_node_ids = kmeans_cut(graph.x, num_subgraphs)
        subgraph_node_ids = {
            key: torch.tensor(node_ids, dtype=torch.int64, device=dev)
            for key, node_ids in subgraph_node_ids.items()
        }
    elif method == "metis":
        subgraph_node_ids = metis_cut(graph.edge_index, graph.num_nodes, num_subgraphs)

    subgraphs = create_comm_indexes(graph, subgraph_node_ids, num_hops=num_hops)

    return subgraphs