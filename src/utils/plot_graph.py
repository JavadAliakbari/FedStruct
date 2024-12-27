import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, remove_isolated_nodes

# from src.utils.Louvian_networkx2 import find_community


def plot_graph(
    edge_index,
    num_nodes,
    num_classes,
    labels,
    pos=None,
    correctly_classified=None,
    ax=None,
) -> None:
    # edge_index = remove_isolated_nodes(edge_index)[0]
    # graph = Data(edge_index=edge_index, num_nodes=num_nodes)
    if correctly_classified is None:
        correctly_classified = torch.zeros(labels.shape[0], dtype=torch.bool)
    nodes = list(range(num_nodes))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_index.T.tolist())
    if pos is None:
        # pos = nx.spectral_layout(G)
        pos = nx.spring_layout(G)
    pos_ = {}
    for i in range(len(pos)):
        pos_[i] = list(pos[i])
    cmap = plt.get_cmap("gist_rainbow", num_classes)
    options = {
        "font_size": 3,
        # "node_size": 5,
        # "node_color": "white",
        # "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
        # "alpha": 0.5,
    }
    edge_color = np.array([0, 0, 0, 0.1])
    edge_color = np.repeat(edge_color[np.newaxis, :], edge_index.shape[1], axis=0)

    node_size = 5 + 45 * correctly_classified
    nx.draw_networkx(
        G,
        node_color=labels.numpy(),
        edge_color=edge_color,
        node_size=node_size,
        with_labels=False,
        pos=pos_,
        arrows=False,
        cmap=cmap,
        ax=ax,
        **options
    )

    return pos


def create_graph() -> Data:
    edge_index1 = np.array(
        [
            [0, 1],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [2, 6],
            [4, 6],
            [5, 2],
            [0, 4],
            [5, 6],
            [6, 3],
        ]
    )
    edge_index1 = np.concatenate(
        (
            edge_index1,
            edge_index1[:, ::-1],
        ),
        axis=0,
    )

    edge_index2 = np.array(
        [
            [0, 1],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [2, 6],
            [4, 6],
            [5, 2],
            [0, 4],
            [1, 6],
            [0, 3],
            [0, 5],
            [6, 5],
        ]
    )
    edge_index2 = np.concatenate(
        (
            edge_index2,
            edge_index2[:, ::-1],
        ),
        axis=0,
    )

    edge_index2 += 7

    inter_connections = np.array([[0, 8], [1, 11], [12, 5]])

    edge_index = np.concatenate((edge_index1, edge_index2, inter_connections), axis=0)
    edge_index = torch.tensor(edge_index)

    graph = Data(edge_index=edge_index.t().contiguous())

    return graph


if __name__ == "__main__":
    # dataset = Planetoid(root="/tmp/Cora", name="Cora")
    # data = dataset[0]
    data = create_graph()

    # print(G.is_directed())
    # G = to_networkx(data)
    # partition = community_louvain.best_partition(G)

    # community = find_community(data)
    # node_color = [0.5 if node == 1 else 1 for node in community]
    node_color = []

    plot_graph(data, node_color)

    plt.show()
