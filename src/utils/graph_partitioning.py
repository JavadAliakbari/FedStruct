import os
from itertools import cycle
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from sknetwork.clustering import Louvain
from torch_geometric.utils import subgraph
from torch_geometric.utils.convert import to_scipy_sparse_matrix

from src.utils.config_parser import Config
from src.utils.graph import Graph
from src.utils.plot_graph import plot_graph

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
    plot_graph(graph, node_color)
    plt.show()


def find_community(graph: Graph):
    adjacency = to_scipy_sparse_matrix(graph.edge_index)
    louvain = Louvain()
    community = louvain.fit_predict(adjacency)

    return community


def create_community_groups(community_map) -> dict:
    community_groups = defaultdict(list)

    for node_id, community in enumerate(community_map):
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


def assign_nodes_to_subgraphs(community_groups, max_subgraph_nodes):
    subgraph_node_ids = {
        subgraph_id: [] for subgraph_id in range(config.subgraph.num_subgraphs)
    }
    subgraphs = cycle(subgraph_node_ids.keys())
    current_subgraph = next(subgraphs)

    counter = 0

    for community in community_groups.keys():
        while (
            len(subgraph_node_ids[current_subgraph]) + len(community_groups[community])
            >= max_subgraph_nodes + config.subgraph.delta
            or len(subgraph_node_ids[current_subgraph]) >= max_subgraph_nodes
        ):
            current_subgraph = next(subgraphs)
            # define counter to avoid stuck in the loop forever
            counter += 1
            if counter == config.subgraph.num_subgraphs:
                return subgraph_node_ids
        subgraph_node_ids[current_subgraph] += community_groups[community]
        counter = 0

    return subgraph_node_ids


def create_subgraps(graph, subgraph_node_ids: dict):
    subgraphs = []
    for community, subgraph_nodes in subgraph_node_ids.items():
        if not isinstance(subgraph_nodes, torch.Tensor):
            node_ids = torch.tensor(subgraph_nodes)
        else:
            node_ids = subgraph_nodes
        edges = graph.edge_index
        edge_mask = torch.isin(edges[0], node_ids) | torch.isin(edges[1], node_ids)
        edge_index = edges[:, edge_mask]

        all_nodes = torch.unique(edge_index.flatten())
        intra_edges = subgraph(
            node_ids,
            edge_index=edge_index,
        )[0]

        external_nodes = all_nodes[~all_nodes.unsqueeze(1).eq(node_ids).any(1)]

        inter_edges = edge_index[
            :,
            ~edge_index.transpose(-1, 0)
            .unsqueeze(1)
            .eq(intra_edges.transpose(-1, 0))
            .all(-1)
            .any(-1),
        ]

        # all_edges = torch.cat((intra_edges, inter_edges), dim=0)

        node_mask = torch.isin(graph.node_ids, node_ids)
        subgraph_node_ids = graph.node_ids[node_mask]
        if "x" in graph.keys:
            x = graph.x[node_mask]
        else:
            x = None

        if "y" in graph.keys:
            y = graph.y[node_mask]
        else:
            y = None

        if "train_mask" in graph.keys:
            train_mask = graph.train_mask[node_mask]
        else:
            train_mask = None

        if "test_mask" in graph.keys:
            test_mask = graph.test_mask[node_mask]
        else:
            test_mask = None

        if "val_mask" in graph.keys:
            val_mask = graph.val_mask[node_mask]
        else:
            val_mask = None

        subgraph_ = Graph(
            x=x,
            y=y,
            edge_index=intra_edges,
            node_ids=subgraph_node_ids,
            external_nodes=external_nodes,
            inter_edges=inter_edges,
            train_mask=train_mask,
            test_mask=test_mask,
            val_mask=val_mask,
        )
        subgraphs.append(subgraph_)

    return subgraphs


def random_assign(graph, max_subgraph_nodes):
    node_ids = graph.node_ids
    idx = torch.randperm(node_ids.shape[0])
    subgraph_node_ids = {}
    for i in range(config.subgraph.num_subgraphs):
        if i < config.subgraph.num_subgraphs - 1:
            subgraph_node_ids[i] = node_ids[
                idx[i * max_subgraph_nodes : (i + 1) * max_subgraph_nodes]
            ]
        else:
            subgraph_node_ids[i] = node_ids[idx[i * max_subgraph_nodes :]]

    return subgraph_node_ids


def louvain_graph_cut(graph: Graph, random=True):
    max_subgraph_nodes = graph.num_nodes // config.subgraph.num_subgraphs
    if random:
        subgraph_node_ids = random_assign(graph, max_subgraph_nodes)
    else:
        community_map = find_community(graph)

        community_groups = create_community_groups(community_map=community_map)

        group_len_max = (
            graph.num_nodes // config.subgraph.num_subgraphs + config.subgraph.delta
        )

        community_groups = make_groups_smaller_than_max(community_groups, group_len_max)

        sorted_community_groups = {
            k: v
            for k, v in sorted(
                community_groups.items(), key=lambda item: len(item[1]), reverse=True
            )
        }

        subgraph_node_ids = assign_nodes_to_subgraphs(
            sorted_community_groups, max_subgraph_nodes
        )

    subgraphs = create_subgraps(graph, subgraph_node_ids)

    return subgraphs
