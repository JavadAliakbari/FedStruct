import os
import json
import random

import torch
import numpy as np
from matplotlib import pyplot as plt

from src.utils.utils import log_config
from src.GNN_server import GNNServer
from src.MLP_server import MLPServer
from src.fedsage_server import FedSAGEServer
from src.utils.logger import get_logger
from src.utils.config_parser import Config
from src.define_graph import define_graph
from src.utils.graph_partitioning import (
    create_mend_graph,
    create_mend_graph2,
    partition_graph,
)

# Change plot canvas size
plt.rcParams["figure.figsize"] = [24, 16]
plt.rcParams["figure.dpi"] = 100  # 200 e.g. is really fine, but slower
plt.rcParams.update({"figure.max_open_warning": 0})

seed = 65
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

path = os.environ.get("CONFIG_PATH")
config = Config(path)


def set_up_system(save_path="./"):
    _LOGGER = get_logger(
        name=f"accuracy_{config.dataset.dataset_name}_{config.structure_model.structure_type}",
        log_on_file=True,
        save_path=save_path,
    )

    log_config(_LOGGER, config)

    graph, num_classes = define_graph(config.dataset.dataset_name)

    if config.model.propagate_type == "DGCN":
        # graph.obtain_a(config.structure_model.DGCN_layers)
        graph.obtain_a(
            config.structure_model.DGCN_layers,
            estimate=config.structure_model.estimate,
            pruning=config.subgraph.prune,
        )

    graph.add_masks(
        train_size=config.subgraph.train_ratio,
        test_size=config.subgraph.test_ratio,
    )

    subgraphs = partition_graph(
        graph, config.subgraph.num_subgraphs, config.subgraph.partitioning
    )

    MLP_server = MLPServer(graph, num_classes, save_path=save_path, logger=_LOGGER)

    for subgraph in subgraphs:
        MLP_server.add_client(subgraph)

    GNN_server = GNNServer(graph, num_classes, save_path=save_path, logger=_LOGGER)

    for subgraph in subgraphs:
        GNN_server.add_client(subgraph)

    GNN_server2 = GNNServer(graph, num_classes, save_path=save_path, logger=_LOGGER)
    for subgraph in subgraphs:
        mend_graph = create_mend_graph(subgraph, graph, 0)
        GNN_server2.add_client(mend_graph)
    GNN_server3 = GNNServer(graph, num_classes, save_path=save_path, logger=_LOGGER)
    for subgraph in subgraphs:
        mend_graph = create_mend_graph(subgraph, graph, 1)
        GNN_server3.add_client(mend_graph)

    FedSage_server = FedSAGEServer(
        graph, num_classes, save_path=save_path, logger=_LOGGER
    )

    for subgraph in subgraphs:
        FedSage_server.add_client(subgraph)

    results = {}

    _LOGGER.info("MLP")
    results[f"server MLP"] = MLP_server.train_local_model()["Test Acc"]
    results[f"local MLP"] = MLP_server.joint_train_g(FL=False)["Average"]["Test Acc"]
    results[f"flga MLP"] = MLP_server.joint_train_g(FL=True)["Average"]["Test Acc"]
    results[f"flwa MLP"] = MLP_server.joint_train_w(FL=True)["Average"]["Test Acc"]

    _LOGGER.info("GNN")
    results[f"server GNN"] = GNN_server.train_local_model()["Test Acc"]
    results[f"local GNN"] = GNN_server.joint_train_g(structure=False, FL=False)[
        "Average"
    ]["Test Acc"]
    results[f"flga GNN"] = GNN_server.joint_train_g(structure=False, FL=True)[
        "Average"
    ]["Test Acc"]
    results[f"flwa GNN"] = GNN_server.joint_train_w(structure=False, FL=True)[
        "Average"
    ]["Test Acc"]
    results[f"sdga GNN"] = GNN_server.joint_train_g(structure=True, FL=True)["Average"][
        "Test Acc"
    ]
    results[f"sdwa GNN"] = GNN_server.joint_train_w(structure=True, FL=True)["Average"][
        "Test Acc"
    ]

    res = FedSage_server.train_fedSage_plus()
    results[f"fedsage WA"] = res["WA"]["Average"]["Test Acc"]
    results[f"fedsage GA"] = res["GA"]["Average"]["Test Acc"]

    _LOGGER.info(json.dumps(results, indent=4))


if __name__ == "__main__":
    save_path = (
        "./results/"
        f"{config.dataset.dataset_name}/"
        f"{config.structure_model.structure_type}/"
        f"{config.subgraph.partitioning}/"
        f"{config.model.propagate_type}/all/"
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    set_up_system(save_path)
    # plt.show()
