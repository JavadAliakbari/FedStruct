import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch_geometric.nn.models import Node2Vec
from torch_geometric.datasets import TUDataset, Planetoid, WikipediaNetwork
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, to_undirected, remove_self_loops
from sklearn.preprocessing import StandardScaler
from src.models.Node2Vec import find_node2vect_embedings


from src.server import Server
from src.utils.GDV import GDV
from src.utils.create_graph import create_heterophilic_graph2, create_homophilic_graph2
from src.utils.graph import Graph
from src.utils.logger import get_logger
from src.utils import config
from src.models.GNN_models import MLP

np.random.seed(4)
torch.manual_seed(4)


def plot(x):
    tsne_model = TSNE(n_components=2, random_state=4, perplexity=25, n_jobs=7)
    x_embed = tsne_model.fit_transform(x)
    cmap = plt.get_cmap("gist_rainbow", num_classes)
    colors = [cmap(1.0 * i / num_classes) for i in range(num_classes)]

    fig, ax = plt.subplots()
    for i in range(num_classes):
        mask = y == i
        class_points = x_embed[mask]
        ax.scatter(
            class_points[:, 0],
            class_points[:, 1],
            color=colors[i],
            marker=f"${i}$",
            label=i,
        )

    ax.legend(loc="lower right")


if __name__ == "__main__":
    _LOGGER = get_logger(
        name=f"SD_{config.dataset}_{config.structure_type}", log_on_file=True
    )

    # num_edges = int(num_nodes * mean_degree / 2)
    # num_patterns = int(0.08 * num_nodes)

    dataset = Planetoid(root=f"/tmp/{config.dataset}", name=config.dataset)
    # dataset = WikipediaNetwork(
    #     root=f"/tmp/{config.dataset}", geom_gcn_preprocess=True, name=config.dataset
    # )

    node_ids = torch.arange(dataset[0].num_nodes)
    edge_index = to_undirected(dataset[0].edge_index)
    edge_index = remove_self_loops(edge_index)[0]
    graph = Graph(
        # x=dataset[0].x,
        y=dataset[0].y,
        edge_index=edge_index,
        node_ids=node_ids,
    )

    # graph = create_heterophilic_graph(num_nodes, num_edges, num_patterns)
    # num_patterns = 50
    # graph = create_homophilic_graph2(num_patterns)
    # graph = create_heterophilic_graph2(num_patterns)
    graph.add_masks()

    y = graph.y.long()
    num_classes = max(y).item() + 1
    server = Server(graph, num_classes, logger=_LOGGER)

    server.train_sd_predictor(epochs=100, plot=True, predict=True)
    server.test_sd_predictor()

    x = server.sd_predictor.graph.structural_features
    graph.x = x
    graph.num_features = x.shape[1]

    server.train_local_mlp()
    _LOGGER.info(f"Server MLP test accuracy: {server.test_local_mlp()}")
    server.train_local_gnn()
    _LOGGER.info(f"Server GNN test accuracy: {server.test_local_gnn()}")

    # x = server.get_sd_embeddings()

    message_passing = MessagePassing(aggr="max")
    for i in range(20):
        cls = MLP(layer_sizes=[config.num_structural_features, 64, num_classes])
        masks = (graph.train_mask, graph.val_mask, graph.test_mask)
        vall_acc, val_loss = cls.fit(x, y, masks, epochs=100, verbose=False)
        _LOGGER.info(f"epoch: {i} vall accuracy: {vall_acc}")
        x = message_passing.propagate(graph.edge_index, x=x)

    plot(x)

    plt.show()
