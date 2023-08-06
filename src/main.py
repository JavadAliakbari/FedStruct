import random
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.datasets import (
    TUDataset,
    Planetoid,
    HeterophilousGraphDataset,
    WikipediaNetwork,
)
from torch_geometric.utils import degree, to_undirected, remove_self_loops

from src.utils import config
from src.utils.create_graph import create_homophilic_graph2
from src.utils.logger import get_logger
from src.utils.graph_partinioning import louvain_graph_cut
from src.server import Server
from src.utils.graph import Graph
from src.utils.plot_graph import plot_graph

# Change plot canvas size
plt.rcParams["figure.figsize"] = [24, 16]
plt.rcParams["figure.dpi"] = 100  # 200 e.g. is really fine, but slower
plt.rcParams.update({"figure.max_open_warning": 0})

random.seed(4)
np.random.seed(4)
torch.manual_seed(4)


def set_up_system():
    _LOGGER = get_logger(name=config.dataset, log_on_file=True)

    try:
        dataset = Planetoid(root=f"/tmp/{config.dataset}", name=config.dataset)
        # dataset = WikipediaNetwork(
        #     root=f"/tmp/{config.dataset}", geom_gcn_preprocess=True, name=config.dataset
        # )
        # dataset = HeterophilousGraphDataset(
        #     root=f"/tmp/{config.dataset}", name=config.dataset
        # )
    except:
        _LOGGER.info("dataset name does not exist!")
        # return

    num_classes = dataset.num_classes
    node_ids = torch.arange(dataset[0].num_nodes)
    edge_index = to_undirected(dataset[0].edge_index)
    edge_index = remove_self_loops(edge_index)[0]

    graph = Graph(
        x=dataset[0].x,
        y=dataset[0].y,
        edge_index=edge_index,
        node_ids=node_ids,
    )

    # num_patterns = 100
    # graph = create_homophilic_graph2(num_patterns)
    # num_classes = max(graph.y).item() + 1

    graph.add_masks(train_size=0.5, test_size=0.2)

    # plot_graph(graph.edge_index, graph.num_nodes)
    # plt.show()

    subgraphs = louvain_graph_cut(graph)

    server = Server(graph, num_classes, logger=_LOGGER)

    for subgraph in subgraphs:
        server.add_client(subgraph)

    _LOGGER.info("MLP")
    server.train_local_mlp()
    _LOGGER.info(f"Server test accuracy: {server.test_local_mlp():.4f}")
    server.train_local_mlps()
    server.train_FLSW(config.epoch_classifier, model_type="MLP")
    server.train_FLSG_MLP(3 * config.epoch_classifier)

    _LOGGER.info("GNN")
    server.train_local_gnn()
    _LOGGER.info(f"Server test accuracy: {server.test_local_gnn():0.4f}")
    server.train_local_gnns()
    server.train_FLSW()
    server.train_FLSG(3 * config.epoch_classifier)
    server.train_SDSW(500)
    server.train_SDSG(500)

    # server.train_sd_ptor()


set_up_system()
# plt.show()
