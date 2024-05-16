import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, remove_isolated_nodes

# from src.utils.Louvian_networkx2 import find_community


def plot_graph(edge_index, node_color=[], pos=None) -> None:
    # edge_index = remove_isolated_nodes(edge_index)[0]
    # graph = Data(edge_index=edge_index, num_nodes=num_nodes)
    G = nx.Graph(edge_index.T.tolist())
    if pos is None:
        pos = nx.spectral_layout(G)
    # options = {
    #     "font_size": 1,
    #     "node_size": 5,
    #     # "node_color": "white",
    #     "edgecolors": "black",
    #     "linewidths": 5,
    #     "width": 1,
    # }

    nx.draw_networkx(G, node_color=node_color, pos=pos, arrows=True)
    # nx.draw_networkx(G, node_color=node_color, pos=pos, arrows=False, **options)

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

    # community = find_community(data)
    # node_color = [0.5 if node == 1 else 1 for node in community]
    node_color = []

    plot_graph(data, node_color)

    plt.show()
