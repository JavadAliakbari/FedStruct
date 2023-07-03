import torch
from matplotlib import pyplot as plt
from torch_geometric.datasets import TUDataset, Planetoid

from src.utils import config
from src.utils.logger import get_logger
from src.utils.graph_partinioning import louvain_graph_cut
from src.server import Server
from src.utils.graph import Graph

# Change plot canvas size
plt.rcParams["figure.figsize"] = [24, 16]
plt.rcParams["figure.dpi"] = 100  # 200 e.g. is really fine, but slower
plt.rcParams.update({"figure.max_open_warning": 0})


def set_up_system():
    _LOGGER = get_logger(name=config.dataset, log_on_file=True)

    if config.dataset == "Cora":
        dataset = Planetoid(root="/tmp/Cora", name="Cora")
    elif config.dataset == "CiteSeer":
        dataset = Planetoid(root="/tmp/CiteSeer", name="CiteSeer")
    elif config.dataset == "PubMed":
        dataset = Planetoid(root="/tmp/PubMed", name="PubMed")
    else:
        _LOGGER.info("dataset name does not exist!")
        return

    node_ids = torch.arange(dataset[0].num_nodes)
    graph = Graph(
        x=dataset[0].x,
        y=dataset[0].y,
        edge_index=dataset[0].edge_index,
        node_ids=node_ids,
    )

    graph.add_masks(train_size=0.5, test_size=0.2)

    num_classes = dataset.num_classes

    subgraphs = louvain_graph_cut(graph)

    server = Server(graph, num_classes, logger=_LOGGER)

    for subgraph in subgraphs:
        server.add_client(subgraph)

    server.joint_train(200)

    # server.train_sd_predictor()
    # server.train_local_classifier()
    # _LOGGER.info(f"Server test accuracy: {server.test_local_classifier()}")
    # print(f"Server test accuracy: {server.test_local_classifier()}")
    # # server.train_Neighbor_predictor_classifier()

    # server.train_local_classifiers()
    # server.train_fedSage()


set_up_system()
plt.show()
