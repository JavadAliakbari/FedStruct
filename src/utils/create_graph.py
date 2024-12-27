from itertools import chain
import math
from matplotlib import pyplot as plt

import numpy as np
import torch
from torch_geometric.utils import to_undirected, remove_self_loops, degree

from src.utils.graph import Graph
from src.utils.plot_graph import plot_graph

np.random.seed(4)

POS = [
    [0, 2],
    [1, 1],
    [1, -1],
    [-1, -1],
    [-1, 1],
    [0, 3],
    [2, 2],
    [2, -0.5],
    [2, -1.5],
    [-2, -0.5],
    [-2, 0.5],
    [-2, 1.5],
]
# POS = {
#     0: np.array([0, 2]),
#     1: np.array([1, 1]),
#     2: np.array([1, -1]),
#     3: np.array([-1, -1]),
#     4: np.array([-1, 1]),
#     5: np.array([0, 3]),
#     6: np.array([2, 2]),
#     7: np.array([2, 0]),
#     8: np.array([2, -1]),
#     9: np.array([-2, -1]),
#     10: np.array([-2, 0]),
#     11: np.array([-2, 2]),
# }


def create_pattern(random_feature):
    num_main_nodes = 5
    edge_index = [
        [0, 1],
        [0, 4],
        [0, 5],
        [1, 2],
        [1, 4],
        [1, 6],
        [2, 3],
        [2, 7],
        [2, 8],
        [3, 4],
        [3, 9],
        [4, 10],
        [4, 11],
    ]

    # l2 = [[n2, n1] for n1, n2 in l1]

    # edge_index = l1

    node_ids = np.array(list(set(chain.from_iterable(edge_index))), dtype=int)
    num_nodes = len(node_ids)

    # y = [True, False, True, False, True] + (num_nodes - 5) * [False]
    # y = [1, 2, 3, 4, 5] + (num_nodes - 5) * [6]
    # y = list(range(1, 13))
    y = list(range(12))
    y = torch.tensor(np.array(y), dtype=torch.long)

    if random_feature == 0:
        x = torch.tensor(np.array([1, 0]), dtype=torch.float32)
        # y += 2
    else:
        x = torch.tensor(np.array([0, 1]), dtype=torch.float32)
        y += 12

    pos = POS

    lock = num_main_nodes * [True] + (num_nodes - num_main_nodes) * [False]
    lock = torch.tensor(np.array(lock))

    return edge_index, node_ids, x, y, lock, pos


def create_pattern1(random_feature=0):
    num_main_nodes = 5
    edge_index = [
        [0, 1],
        [0, 4],
        [0, 5],
        [1, 2],
        [1, 4],
        [1, 6],
        [2, 3],
        [2, 7],
        [2, 8],
        [3, 4],
        [3, 9],
        [4, 10],
        [4, 11],
    ]

    # l2 = [[n2, n1] for n1, n2 in l1]

    # edge_index = l1

    node_ids = np.array(list(set(chain.from_iterable(edge_index))), dtype=int)
    num_nodes = len(node_ids)
    if random_feature == 0:
        x = torch.tensor(np.array([1, 0]), dtype=torch.float32)
        label_value = 1
    else:
        x = torch.tensor(np.array([0, 1]), dtype=torch.float32)
        label_value = 3
    # y = [True, False, True, False, True] + (num_nodes - 5) * [False]
    y = 5 * [label_value] + (num_nodes - 5) * [0]
    # y = [1, 2, 3, 4, 5] + (num_nodes - 5) * [6]
    # y = list(range(1, 13))
    y = torch.tensor(np.array(y), dtype=torch.long)

    pos = POS

    lock = num_main_nodes * [True] + (num_nodes - num_main_nodes) * [False]
    lock = torch.tensor(np.array(lock))

    return edge_index, node_ids, x, y, lock, pos


def create_pattern2(random_feature=0):
    num_main_nodes = 5
    edge_index = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [1, 2],
        # [1, 4],
        [1, 6],
        [2, 3],
        [2, 7],
        [2, 8],
        # [3, 4],
        [3, 9],
        [4, 10],
        [4, 11],
    ]

    # l2 = [[n2, n1] for n1, n2 in l1]

    # edge_index = l1

    node_ids = np.array(list(set(chain.from_iterable(edge_index))), dtype=int)
    num_nodes = len(node_ids)

    if random_feature == 0:
        x = torch.tensor(np.array([1, 0]), dtype=torch.float32)
        label_value = 2
    else:
        x = torch.tensor(np.array([0, 1]), dtype=torch.float32)
        label_value = 4

    y = 5 * [label_value] + (num_nodes - 5) * [0]
    # y = [1, 2, 3, 4, 5] + (num_nodes - 5) * [6]
    # y = list(range(1, 13))
    y = torch.tensor(np.array(y), dtype=torch.long)

    pos = POS

    lock = num_main_nodes * [True] + (num_nodes - num_main_nodes) * [False]
    lock = torch.tensor(np.array(lock))

    return edge_index, node_ids, x, y, lock, pos


def is_connected(edge_index, node_ids):
    num_nodes = len(node_ids)

    # connection_mask = np.array([True] + (num_nodes - 1) * [False])
    connectesd_nodes = node_ids[np.array([0])]

    count = 0
    while len(connectesd_nodes) < num_nodes:
        edge_mask = edge_index.unsqueeze(2).eq(connectesd_nodes).any(2).any(0)
        new_conncted_nodes = np.unique(edge_index[:, edge_mask].flatten())

        if len(new_conncted_nodes) == len(connectesd_nodes):
            # print("\ndidn't find")
            return False
        else:
            connectesd_nodes = new_conncted_nodes

        count += 1
        # print(count, end="\r")

    return True


def circular_transform(pos):
    x_min = min(pos[:, 0])
    x_max = max(pos[:, 0]) + 2
    y_min = min(pos[:, 1])
    y_max = max(pos[:, 1])

    normalize_pos = np.vstack(
        ((pos[:, 0] - x_min) / (x_max - x_min), (pos[:, 1] - y_min) / (y_max - y_min))
    ).T

    r_min = 1
    r_max = 2
    r = (r_max - r_min) * normalize_pos[:, 1] + r_min
    teta = 2 * math.pi * normalize_pos[:, 0] - math.pi / 2
    new_pos = np.vstack((r * np.cos(teta), -r * np.sin(teta))).T

    return new_pos


def create_heterophilic_graph(num_nodes, num_edges, num_patterns):
    offset = 0
    base_edge_index = []
    node_ids = torch.arange(num_nodes)
    # y = num_nodes * [False]
    y = torch.zeros(num_nodes, dtype=torch.long)
    # lock = num_nodes * [False]
    lock = np.zeros(num_nodes, dtype=bool)
    for _ in range(num_patterns):
        edge_index_, node_ids_, y_, lock_, pos_ = create_pattern()

        edge_index_ = list(map(lambda x: [x[0] + offset, x[1] + offset], edge_index_))
        base_edge_index += edge_index_

        # node_ids_ = list(map(lambda x: x + offset, node_ids_))
        node_ids_ += offset

        # list(map(y.__setitem__, node_ids_, y_))
        # list(map(lock.__setitem__, node_ids_, lock_))
        y[node_ids_] = y_
        lock[node_ids_] = lock_

        offset += len(node_ids_)

    free_nodes = node_ids[~lock]
    pp = False
    while not pp:
        edge_index = torch.tensor(
            base_edge_index
            + np.random.choice(
                free_nodes, (num_edges - len(base_edge_index), 2), replace=True
            ).tolist()
        ).T

        pp = is_connected(edge_index, node_ids)

    edge_index = remove_self_loops(edge_index)[0]
    edge_index = to_undirected(edge_index)

    while edge_index.shape[1] < 2 * num_edges:
        new_edges = np.random.choice(
            free_nodes,
            (2, int(num_edges - edge_index.shape[1] / 2)),
            replace=True,
        )
        edge_index = torch.hstack(
            (
                edge_index,
                torch.tensor(new_edges),
            ),
        )

        edge_index = remove_self_loops(edge_index)[0]
        edge_index = to_undirected(edge_index)

    graph = Graph(edge_index=edge_index, node_ids=node_ids, y=y)

    return graph


def create_heterophilic_graph2(
    num_patterns, circular_pos=False, use_random_features=False
):
    num_nodes = 12 * num_patterns
    offset = 0
    base_edge_index = []

    node_ids = torch.arange(num_nodes)
    x = torch.zeros((num_nodes, 2), dtype=torch.float32)
    # y = num_nodes * [False]
    y = torch.zeros(num_nodes, dtype=torch.long)
    # lock = num_nodes * [False]
    lock = np.zeros(num_nodes, dtype=bool)
    pos = []

    random_feature = 0
    pos_offset = 0
    for _ in range(num_patterns):
        if use_random_features:
            random_feature = np.random.randint(0, 2)
        edge_index_, node_ids_, x_, y_, lock_, pos_ = create_pattern(random_feature)

        edge_index_ = list(map(lambda u: [u[0] + offset, u[1] + offset], edge_index_))
        base_edge_index += edge_index_

        # node_ids_ = list(map(lambda x: x + offset, node_ids_))
        node_ids_ += offset

        # list(map(y.__setitem__, node_ids_, y_))
        # list(map(lock.__setitem__, node_ids_, lock_))
        y[node_ids_] = y_
        x[node_ids_] = x_
        lock[node_ids_] = lock_

        pos_ = list(map(lambda x: [x[0] + pos_offset, x[1]], pos_))
        pos += pos_
        pos_offset += 6

        offset += len(node_ids_)

    n = list(range(num_patterns - 1))
    n2 = list(range(num_patterns - 2))
    n2_c = list(range(2))

    edge_index_1 = list(map(lambda i: [30 + 12 * i, 11 + 12 * i], n2)) + list(
        map(lambda i: [6 + 12 * i, 11 + 12 * (i + num_patterns - 2)], n2_c)
    )
    edge_index_2 = list(map(lambda i: [22 + 12 * i, 7 + 12 * i], n)) + [
        [
            10,
            7 + 12 * (num_patterns - 1),
        ]
    ]
    edge_index_3 = list(map(lambda i: [21 + 12 * i, 8 + 12 * i], n)) + [
        [
            9,
            8 + 12 * (num_patterns - 1),
        ]
    ]
    edge_index_4 = list(map(lambda i: [29 + 12 * i, 5 + 12 * i], n2)) + list(
        map(lambda i: [5 + 12 * i, 5 + 12 * (i + num_patterns - 2)], n2_c)
    )

    base_edge_index += edge_index_1
    base_edge_index += edge_index_2
    base_edge_index += edge_index_3
    base_edge_index += edge_index_4

    edge_index = torch.tensor(np.array(base_edge_index).T)
    edge_index = remove_self_loops(edge_index)[0]
    edge_index = to_undirected(edge_index)

    pos = np.array(pos, dtype=float)

    if circular_pos:
        pos = circular_transform(pos)
    pos = {key: val for key, val in zip(node_ids.numpy(), pos)}

    graph = Graph(edge_index=edge_index, node_ids=node_ids, x=x, y=y, pos=pos)

    return graph


def create_homophilic_graph(num_nodes, num_edges, num_patterns):
    offset = 0
    base_edge_index = []
    node_ids = torch.arange(num_nodes)
    # y = num_nodes * [False]
    y = torch.zeros(num_nodes, dtype=torch.long)
    # lock = num_nodes * [False]
    lock = np.zeros(num_nodes, dtype=bool)
    for _ in range(num_patterns):
        random_number = np.random.randint(0, 2)
        if random_number == 0:
            edge_index_, node_ids_, y_, lock_ = create_pattern1()
        else:
            edge_index_, node_ids_, y_, x_, lock_ = create_pattern2()

        edge_index_ = list(map(lambda x: [x[0] + offset, x[1] + offset], edge_index_))
        base_edge_index += edge_index_

        # node_ids_ = list(map(lambda x: x + offset, node_ids_))
        node_ids_ += offset

        # list(map(y.__setitem__, node_ids_, y_))
        # list(map(lock.__setitem__, node_ids_, lock_))
        y[node_ids_] = y_
        lock[node_ids_] = lock_

        offset += len(node_ids_)

    free_nodes = node_ids[~lock]
    pp = False
    while not pp:
        edge_index = torch.tensor(
            base_edge_index
            + np.random.choice(
                free_nodes, (num_edges - len(base_edge_index), 2), replace=True
            ).tolist()
        ).T

        pp = is_connected(edge_index, node_ids)

    edge_index = remove_self_loops(edge_index)[0]
    edge_index = to_undirected(edge_index)

    while edge_index.shape[1] < 2 * num_edges:
        new_edges = np.random.choice(
            free_nodes,
            (2, int(num_edges - edge_index.shape[1] / 2)),
            replace=True,
        )
        edge_index = torch.hstack(
            (
                edge_index,
                torch.tensor(new_edges),
            ),
        )

        edge_index = remove_self_loops(edge_index)[0]
        edge_index = to_undirected(edge_index)

    graph = Graph(edge_index=edge_index, node_ids=node_ids, y=y)

    return graph


def create_homophilic_graph2(
    num_patterns, circular_pos=False, use_random_features=False
):
    num_nodes = 12 * num_patterns
    offset = 0
    base_edge_index = []
    node_ids = torch.arange(num_nodes)
    x = torch.zeros((num_nodes, 2), dtype=torch.float32)
    # y = num_nodes * [False]
    y = torch.zeros(num_nodes, dtype=torch.long)
    # lock = num_nodes * [False]
    lock = np.zeros(num_nodes, dtype=bool)
    pos = []
    pos_offset = 0
    random_feature = 0
    for _ in range(num_patterns):
        random_structure = np.random.randint(0, 2)
        if use_random_features:
            random_feature = np.random.randint(0, 2)
        if random_structure == 0:
            edge_index_, node_ids_, x_, y_, lock_, pos_ = create_pattern1(
                random_feature
            )
        else:
            edge_index_, node_ids_, x_, y_, lock_, pos_ = create_pattern2(
                random_feature
            )

        edge_index_ = list(map(lambda x: [x[0] + offset, x[1] + offset], edge_index_))
        base_edge_index += edge_index_

        # node_ids_ = list(map(lambda x: x + offset, node_ids_))
        node_ids_ += offset

        # list(map(y.__setitem__, node_ids_, y_))
        # list(map(lock.__setitem__, node_ids_, lock_))
        x[node_ids_] = x_
        y[node_ids_] = y_
        lock[node_ids_] = lock_

        pos_ = list(map(lambda x: [x[0] + pos_offset, x[1]], pos_))
        pos += pos_
        pos_offset += 6

        offset += len(node_ids_)

    n1 = list(range(num_patterns - 1))
    n1_c = [0]
    n2 = list(range(num_patterns - 2))
    n2_c = [0, 1]

    edge_index_1 = list(map(lambda i: [30 + 12 * i, 11 + 12 * i], n2)) + list(
        map(lambda i: [6 + 12 * i, 11 + 12 * (i + num_patterns - 2)], n2_c)
    )
    edge_index_2 = list(map(lambda i: [22 + 12 * i, 7 + 12 * i], n1)) + [
        [10, 7 + 12 * (num_patterns - 1)]
    ]
    edge_index_3 = list(map(lambda i: [21 + 12 * i, 8 + 12 * i], n1)) + [
        [9, 8 + 12 * (num_patterns - 1)]
    ]
    edge_index_4 = list(map(lambda i: [29 + 12 * i, 5 + 12 * i], n2)) + list(
        map(lambda i: [5 + 12 * i, 5 + 12 * (i + num_patterns - 2)], n2_c)
    )

    base_edge_index += edge_index_1
    base_edge_index += edge_index_2
    base_edge_index += edge_index_3
    base_edge_index += edge_index_4

    edge_index = torch.tensor(np.array(base_edge_index).T)
    edge_index = remove_self_loops(edge_index)[0]
    edge_index = to_undirected(edge_index)

    pos = np.array(pos, dtype=float)

    if circular_pos:
        pos = circular_transform(pos)

    pos = {key: val for key, val in zip(node_ids.numpy(), pos)}

    graph = Graph(edge_index=edge_index, node_ids=node_ids, x=x, y=y, pos=pos)

    return graph


if __name__ == "__main__":
    # num_nodes = 1000
    # mean_degree = 5
    # num_edges = int(num_nodes * mean_degree / 2)
    # num_patterns = int(0.05 * num_nodes)

    # graph = create_heterophilic_graph(num_nodes, num_edges, num_patterns)

    graph = create_homophilic_graph2(num_patterns=5, circular_pos=True)
    y = graph.y.long()
    num_classes = graph.num_classes
    cmap = plt.get_cmap("gist_rainbow", num_classes)
    colors = [cmap(1.0 * i / num_classes) for i in range(num_classes)]
    node_color = [colors[i] for i in y]
    plot_graph(
        graph.edge_index,
        graph.num_nodes,
        node_color=node_color,
        pos=graph.pos,
    )
    plt.show()
    a = 2
