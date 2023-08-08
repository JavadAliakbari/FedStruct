import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from sklearn.manifold import TSNE
from torch_geometric.datasets import TUDataset, Planetoid, WikipediaNetwork
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, to_undirected, remove_self_loops


from src.server import Server
from src.utils.create_graph import create_heterophilic_graph2, create_homophilic_graph2
from src.utils.graph import Graph
from src.utils.logger import get_logger
from src.utils import config
from src.models.GNN_models import MLP

np.random.seed(4)
torch.manual_seed(4)


def plot(x):
    tsne_model = TSNE(n_components=2, random_state=4, perplexity=20, n_jobs=7)
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
    # dataset = Planetoid(root=f"/tmp/{config.dataset}", name=config.dataset)
    # dataset = WikipediaNetwork(
    #     root=f"/tmp/{config.dataset}", geom_gcn_preprocess=True, name=config.dataset
    # )

    # node_ids = torch.arange(dataset[0].num_nodes)
    # edge_index = to_undirected(dataset[0].edge_index)
    # edge_index = remove_self_loops(edge_index)[0]
    # graph = Graph(
    #     y=dataset[0].y,
    #     edge_index=edge_index,
    #     node_ids=node_ids,
    # )
    # _LOGGER = get_logger(
    #     name=f"SD_{config.dataset}_{config.structure_type}", log_on_file=True
    # )

    num_patterns = 50
    graph = create_homophilic_graph2(num_patterns)
    # graph = create_heterophilic_graph2(num_patterns)
    _LOGGER = get_logger(name=f"SD_Costum_{config.structure_type}", log_on_file=True)

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

    message_passing = MessagePassing(aggr="mean")
    for i in range(20):
        cls = MLP(layer_sizes=[config.num_structural_features, 64, num_classes])
        masks = (graph.train_mask, graph.val_mask, graph.test_mask)
        vall_acc, val_loss = cls.fit(x, y, masks, epochs=100, verbose=False)
        _LOGGER.info(f"epoch: {i} vall accuracy: {vall_acc}")
        x = message_passing.propagate(graph.edge_index, x=x)

    x = server.sd_predictor.graph.structural_features
    plot(x)

    plt.show()
