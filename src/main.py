import os
import json
import random
from datetime import datetime

now = datetime.now().strftime("%Y%m%d_%H%M%S")
os.environ["now"] = now

import torch
import numpy as np

from src.FedPub.fedpub_server import FedPubServer
from src.utils.utils import log_config
from src.GNN.GNN_server import GNNServer
from src.MLP.MLP_server import MLPServer
from src.fedsage.fedsage_server import FedSAGEServer
from src.utils.logger import get_logger
from src.utils.config_parser import Config
from src.utils.define_graph import define_graph
from src.utils.graph_partitioning import (
    create_mend_graph,
    create_mend_graph2,
    partition_graph,
)

seed = 65
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

path = os.environ.get("CONFIG_PATH")
config = Config(path)


def set_up_system(save_path="./"):
    _LOGGER = get_logger(
        name=f"{now}_{config.dataset.dataset_name}",
        log_on_file=True,
        save_path=save_path,
    )

    log_config(_LOGGER, config)

    graph = define_graph(config.dataset.dataset_name)

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

    MLP_server = MLPServer(graph, save_path=save_path, logger=_LOGGER)
    for subgraph in subgraphs:
        MLP_server.add_client(subgraph)

    GNN_server = GNNServer(graph, save_path=save_path, logger=_LOGGER)
    for subgraph in subgraphs:
        GNN_server.add_client(subgraph)

    GNN_server2 = GNNServer(graph, save_path=save_path, logger=_LOGGER)
    for subgraph in subgraphs:
        mend_graph = create_mend_graph(subgraph, graph, 0)
        GNN_server2.add_client(mend_graph)

    GNN_server3 = GNNServer(graph, save_path=save_path, logger=_LOGGER)
    for subgraph in subgraphs:
        mend_graph = create_mend_graph(subgraph, graph, 1)
        GNN_server3.add_client(mend_graph)

    fedsage_server = FedSAGEServer(graph, save_path=save_path, logger=_LOGGER)
    for subgraph in subgraphs:
        fedsage_server.add_client(subgraph)

    fedpub_server = FedPubServer(graph, save_path=save_path, logger=_LOGGER)
    for subgraph in subgraphs:
        fedpub_server.add_client(subgraph)

    results = {}

    # _LOGGER.info("MLP")
    # results[f"server MLP"] = MLP_server.train_local_model()["Test Acc"]
    # results[f"local MLP"] = MLP_server.joint_train_g(FL=False)["Average"]["Test Acc"]
    # results[f"flga MLP"] = MLP_server.joint_train_g(FL=True)["Average"]["Test Acc"]

    # _LOGGER.info("GNN")
    # results[f"server GNN"] = GNN_server.train_local_model(propagate_type="GNN")[
    #     "Test Acc"
    # ]
    # results[f"local GNN"] = GNN_server.joint_train_g(structure=False, FL=False)[
    #     "Average"
    # ]["Test Acc"]
    # results[f"flga GNN"] = GNN_server.joint_train_g(
    #     propagate_type="GNN", structure=False, FL=True
    # )["Average"]["Test Acc"]
    # results[f"sdga GNN"] = GNN_server.joint_train_g(structure=True, FL=True)["Average"][
    #     "Test Acc"
    # ]

    # res = fedsage_server.train_fedSage_plus()
    # results[f"fedsage WA"] = res["WA"]["Average"]["Test Acc"]
    # results[f"fedsage GA"] = res["GA"]["Average"]["Test Acc"]

    results[f"fedpub"] = fedpub_server.start()["Average"]["Test Acc"]

    _LOGGER.info(json.dumps(results, indent=4))


if __name__ == "__main__":
    save_path = (
        "./results/"
        f"{config.dataset.dataset_name}/"
        f"{config.structure_model.structure_type}/"
        f"{config.subgraph.partitioning}/"
        f"{config.model.propagate_type}/"
        f"{config.subgraph.num_subgraphs}/all/"
    )
    os.makedirs(save_path, exist_ok=True)
    set_up_system(save_path)
    # plt.show()
