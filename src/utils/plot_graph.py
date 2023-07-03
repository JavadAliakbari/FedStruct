import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

# from src.utils.Louvian_networkx2 import find_community


def plot_graph(data, node_color=[]) -> None:
    G = to_networkx(data)
    G.remove_nodes_from(list(nx.isolates(G)))
    options = {
        "font_size": 5,
        "node_size": 50,
        # "node_color": "white",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
    }

    nx.draw_networkx(G, node_color=node_color, **options)

    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")


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

    community = find_community(data)
    node_color = [0.5 if node == 1 else 1 for node in community]
    node_color = []

    plot_graph(data, node_color)

    plt.show()
