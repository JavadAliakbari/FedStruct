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
from torch_geometric.utils import to_undirected, remove_self_loops

from src.server import Server
from src.utils.graph import Graph
from src.utils.logger import get_logger
from src.utils.config_parser import Config
from src.utils.graph_partinioning import louvain_graph_cut
from src.utils.create_graph import create_homophilic_graph2, create_heterophilic_graph2

# Change plot canvas size
plt.rcParams["figure.figsize"] = [24, 16]
plt.rcParams["figure.dpi"] = 100  # 200 e.g. is really fine, but slower
plt.rcParams.update({"figure.max_open_warning": 0})

random.seed(4)
np.random.seed(4)
torch.manual_seed(4)

config = Config()


def log_config(_LOGGER):
    _LOGGER.info(f"dataset name: {config.dataset.dataset_name}")
    _LOGGER.info(f"num subgraphs: {config.subgraph.num_subgraphs}")
    _LOGGER.info(f"batch: {config.model.batch}")
    _LOGGER.info(f"batch size: {config.model.batch_size}")
    _LOGGER.info(f"learning rate: {config.model.lr}")
    _LOGGER.info(f"weight decay: {config.model.weight_decay}")
    _LOGGER.info(f"dropout: {config.model.dropout}")
    _LOGGER.info(f"gnn layer type: {config.model.gnn_layer_type}")
    _LOGGER.info(f"gnn layer sizes: {config.model.gnn_layer_sizes}")
    _LOGGER.info(f"mlp layer sizes: {config.model.mlp_layer_sizes}")
    _LOGGER.info(f"sd ratio: {config.structure_model.sd_ratio}")
    _LOGGER.info(
        f"structure layers size: {config.structure_model.structure_layers_sizes}"
    )
    _LOGGER.info(f"structure type: {config.structure_model.structure_type}")
    _LOGGER.info(
        f"num structural features: {config.structure_model.num_structural_features}"
    )
    _LOGGER.info(f"loss: {config.structure_model.loss}")


def set_up_system():
    save_path = f"./results/{config.dataset.dataset_name}/{config.structure_model.structure_type}/all/"
    _LOGGER = get_logger(
        name=f"accuracy_{config.dataset.dataset_name}_{config.structure_model.structure_type}",
        log_on_file=True,
        save_path=save_path,
    )

    log_config(_LOGGER)

    try:
        dataset = None
        if config.dataset.dataset_name in ["Cora", "PubMed", "CiteSeer"]:
            dataset = Planetoid(
                root=f"/tmp/{config.dataset.dataset_name}",
                name=config.dataset.dataset_name,
            )
            node_ids = torch.arange(dataset[0].num_nodes)
            edge_index = dataset[0].edge_index
            num_classes = dataset.num_classes
        elif config.dataset.dataset_name in ["chameleon", "crocodile", "squirrel"]:
            dataset = WikipediaNetwork(
                root=f"/tmp/{config.dataset.dataset_name}",
                geom_gcn_preprocess=True,
                name=config.dataset.dataset_name,
            )
            node_ids = torch.arange(dataset[0].num_nodes)
            edge_index = dataset[0].edge_index
            num_classes = dataset.num_classes
        elif config.dataset.dataset_name in [
            "Roman-empire",
            "Amazon-ratings",
            "Minesweeper",
            "Tolokers",
            "Questions",
        ]:
            dataset = HeterophilousGraphDataset(
                root=f"/tmp/{config.dataset.dataset_name}",
                name=config.dataset.dataset_name,
            )
        elif config.dataset.dataset_name == "Heterophilic_example":
            num_patterns = 500
            graph = create_heterophilic_graph2(num_patterns, use_random_features=True)
        elif config.dataset.dataset_name == "Homophilic_example":
            num_patterns = 100
            graph = create_homophilic_graph2(num_patterns, use_random_features=True)

    except:
        _LOGGER.info("dataset name does not exist!")
        return

    if dataset is not None:
        node_ids = torch.arange(dataset[0].num_nodes)
        edge_index = dataset[0].edge_index
        num_classes = dataset.num_classes

        edge_index = to_undirected(edge_index)
        edge_index = remove_self_loops(edge_index)[0]
        graph = Graph(
            x=dataset[0].x,
            y=dataset[0].y,
            edge_index=edge_index,
            node_ids=node_ids,
        )
    else:
        num_classes = max(graph.y).item() + 1

    graph.add_masks(train_size=0.5, test_size=0.2)

    subgraphs = louvain_graph_cut(graph)

    MLP_server = Server(
        graph, num_classes, classifier_type="MLP", save_path=save_path, logger=_LOGGER
    )

    for subgraph in subgraphs:
        MLP_server.add_client(subgraph)

    GNN_server = Server(
        graph, num_classes, classifier_type="GNN", save_path=save_path, logger=_LOGGER
    )

    for subgraph in subgraphs:
        GNN_server.add_client(subgraph)

    _LOGGER.info("MLP")
    MLP_server.train_local_classifier(config.model.epoch_classifier)
    _LOGGER.info(f"Server test accuracy: {MLP_server.test_local_classifier():.4f}")
    MLP_server.train_local_classifiers(config.model.epoch_classifier)
    MLP_server.train_FLWA(config.model.epoch_classifier)
    MLP_server.train_FLGA(config.model.epoch_classifier)

    _LOGGER.info("GNN")
    GNN_server.train_local_classifier(config.model.epoch_classifier)
    _LOGGER.info(f"Server test accuracy: {GNN_server.test_local_classifier():0.4f}")
    GNN_server.train_local_classifiers(config.model.epoch_classifier)
    GNN_server.train_FLWA(config.model.epoch_classifier)
    GNN_server.train_FLGA(config.model.epoch_classifier)
    GNN_server.train_SD_Server(config.model.epoch_classifier)
    GNN_server.train_SDWA(config.model.epoch_classifier)
    GNN_server.train_SDGA(config.model.epoch_classifier)

    # GNN_server.train_local_sd(config.model.epoch_classifier)

    # GNN_server.train_locsages()
    # GNN_server.train_fedSage_plus()
    # server.train_sd_ptor()


if __name__ == "__main__":
    set_up_system()
    # plt.show()
