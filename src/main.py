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

    _LOGGER.info("MLP")
    res = MLP_server.train_local_model()
    results[f"server MLP"] = round(res["Test Acc"], 4)

    res = MLP_server.joint_train_g(FL=False)
    results[f"local MLP"] = round(res["Average"]["Test Acc"], 4)

    res = MLP_server.joint_train_g(FL=True)
    results[f"flga MLP"] = round(res["Average"]["Test Acc"], 4)

    _LOGGER.info("GNN")
    res = GNN_server.train_local_model(propagate_type="GNN")
    results[f"server GNN"] = round(res["Test Acc"], 4)

    res = GNN_server.joint_train_g(FL=False)
    results[f"local GNN"] = round(res["Average"]["Test Acc"], 4)

    res = GNN_server.joint_train_g(propagate_type="GNN", FL=True)
    results[f"flga GNN"] = round(res["Average"]["Test Acc"], 4)

    res = GNN_server.joint_train_g(data_type="f+s", FL=True)
    results[f"sdga GNN"] = round(res["Average"]["Test Acc"], 4)

    res = fedsage_server.train_fedSage_plus()
    results[f"fedsage WA"] = round(res["WA"]["Average"]["Test Acc"], 4)
    results[f"fedsage GA"] = round(res["GA"]["Average"]["Test Acc"], 4)

    res = fedpub_server.start()
    results[f"fedpub"] = round(res["Average"]["Test Acc"], 4)

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
