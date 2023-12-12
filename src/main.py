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
from src.GNN_server import GNNServer
from src.MLP_server import MLPServer
from src.fedsage_server import FedSAGEServer

from src.utils.graph import Graph
from src.utils.logger import get_logger
from src.utils.config_parser import Config
from src.utils.graph_partitioning import louvain_graph_cut
from src.utils.create_graph import create_homophilic_graph2, create_heterophilic_graph2

# Change plot canvas size
plt.rcParams["figure.figsize"] = [24, 16]
plt.rcParams["figure.dpi"] = 100  # 200 e.g. is really fine, but slower
plt.rcParams.update({"figure.max_open_warning": 0})

seed = 4
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

config = Config()


def log_config(_LOGGER):
    _LOGGER.info(f"dataset name: {config.dataset.dataset_name}")
    _LOGGER.info(f"num subgraphs: {config.subgraph.num_subgraphs}")
    _LOGGER.info(f"num Epochs: {config.model.epoch_classifier}")
    _LOGGER.info(f"batch: {config.model.batch}")
    _LOGGER.info(f"batch size: {config.model.batch_size}")
    _LOGGER.info(f"learning rate: {config.model.lr}")
    _LOGGER.info(f"weight decay: {config.model.weight_decay}")
    _LOGGER.info(f"dropout: {config.model.dropout}")
    _LOGGER.info(f"gnn layer type: {config.model.gnn_layer_type}")
    _LOGGER.info(f"propagate type: {config.model.propagate_type}")
    _LOGGER.info(f"gnn layer sizes: {config.feature_model.gnn_layer_sizes}")
    _LOGGER.info(f"mlp layer sizes: {config.feature_model.mlp_layer_sizes}")
    _LOGGER.info(f"structure mp layers: {config.structure_model.mp_layers}")
    _LOGGER.info(f"feature mp layers: {config.feature_model.mp_layers}")
    _LOGGER.info(f"sd ratio: {config.structure_model.sd_ratio}")
    if config.model.propagate_type == "GNN":
        _LOGGER.info(
            f"structure layers size: {config.structure_model.GNN_structure_layers_sizes}"
        )
    else:
        _LOGGER.info(
            f"structure layers size: {config.structure_model.MP_structure_layers_sizes}"
        )
    _LOGGER.info(f"structure type: {config.structure_model.structure_type}")
    _LOGGER.info(
        f"num structural features: {config.structure_model.num_structural_features}"
    )
    _LOGGER.info(f"loss: {config.structure_model.loss}")
    _LOGGER.info(
        f"Train-Test ratio: [{config.subgraph.train_ratio}, {config.subgraph.test_ratio}]"
    )


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

    graph.add_masks(
        train_size=config.subgraph.train_ratio,
        test_size=config.subgraph.test_ratio,
    )

    subgraphs = louvain_graph_cut(graph, True)

    MLP_server = MLPServer(graph, num_classes, save_path=save_path, logger=_LOGGER)

    for subgraph in subgraphs:
        MLP_server.add_client(subgraph)

    GNN_server = GNNServer(graph, num_classes, save_path=save_path, logger=_LOGGER)

    for subgraph in subgraphs:
        GNN_server.add_client(subgraph)

    FedSage_server = FedSAGEServer(
        graph, num_classes, save_path=save_path, logger=_LOGGER
    )

    for subgraph in subgraphs:
        FedSage_server.add_client(subgraph)

    _LOGGER.info("MLP")
    MLP_server.train_local_model()
    MLP_server.joint_train_g(FL=False)
    MLP_server.joint_train_g(FL=True)
    MLP_server.joint_train_w(FL=True)

    _LOGGER.info("GNN")
    GNN_server.train_local_model()
    GNN_server.joint_train_g(structure=False, FL=False)
    GNN_server.joint_train_g(structure=False, FL=True)
    GNN_server.joint_train_w(structure=False, FL=True)
    GNN_server.joint_train_g(structure=True, FL=True)
    GNN_server.joint_train_w(structure=True, FL=True)

    FedSage_server.train_locsages()
    FedSage_server.train_fedSage_plus()


if __name__ == "__main__":
    set_up_system()
    # plt.show()
