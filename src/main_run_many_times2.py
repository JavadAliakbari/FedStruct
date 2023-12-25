import os
from copy import deepcopy
import random
import json

import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch_geometric.datasets import (
    TUDataset,
    Planetoid,
    HeterophilousGraphDataset,
    WikipediaNetwork,
)
from torch_geometric.utils import to_undirected, remove_self_loops
from tqdm import tqdm
from src.GNN_server import GNNServer
from src.MLP_server import MLPServer
from src.fedsage_server import FedSAGEServer

from src.server import Server
from src.utils.graph import Graph
from src.utils.logger import get_logger
from src.utils.config_parser import Config
from src.utils.graph_partitioning import louvain_graph_cut
from src.utils.create_graph import create_homophilic_graph2, create_heterophilic_graph2
from src.main import log_config

# Change plot canvas size
plt.rcParams["figure.figsize"] = [24, 16]
plt.rcParams["figure.dpi"] = 100  # 200 e.g. is really fine, but slower
plt.rcParams.update({"figure.max_open_warning": 0})

seed = 4
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

path = os.environ.get("CONFIG_PATH")
config = Config(path)


def define_graph() -> Graph:
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
        # _LOGGER.info("dataset name does not exist!")
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
            keep_sfvs=True,
        )
    else:
        num_classes = max(graph.y).item() + 1

    graph.add_masks(
        train_size=config.subgraph.train_ratio,
        test_size=config.subgraph.test_ratio,
    )
    graph.num_classes = num_classes

    return graph


def save_average_result(average_result):
    final_result = {}
    for key, val in average_result.items():
        if "Average" in val.keys():
            final_result[key] = val["Average"]["Test Acc"]
        else:
            final_result[key] = val["Test Acc"]

    df = pd.DataFrame.from_dict(final_result, orient="index")
    df.to_csv(f"{save_path}final_result.csv")


def run(
    graph: Graph,
    MLP_server: MLPServer,
    GNN_server: GNNServer,
    FedSage_server: FedSAGEServer,
    bar: tqdm,
):
    graph.add_masks(
        train_size=config.subgraph.train_ratio, test_size=config.subgraph.test_ratio
    )

    MLP_server.remove_clients()
    GNN_server.remove_clients()
    FedSage_server.remove_clients()

    subgraphs = louvain_graph_cut(graph, config.subgraph.random)

    for subgraph in subgraphs:
        MLP_server.add_client(subgraph)
        GNN_server.add_client(subgraph)
        FedSage_server.add_client(subgraph)

    result = {}

    res = MLP_server.train_local_model(log=False, plot=False)
    result["server_mlp"] = res
    bar.set_postfix_str(f"server_mlp: {res['Test Acc']}")
    res = MLP_server.joint_train_g(FL=False, log=False, plot=False)
    result["local_mlp"] = res
    bar.set_postfix_str(f"local_mlp: {res['Average']['Test Acc']}")
    res = MLP_server.joint_train_w(FL=True, log=False, plot=False)
    result["flwa_mlp"] = res
    bar.set_postfix_str(f"flwa_mlp: {res['Average']['Test Acc']}")
    res = MLP_server.joint_train_g(FL=True, log=False, plot=False)
    result["flga_mlp"] = res
    bar.set_postfix_str(f"flga_mlp: {res['Average']['Test Acc']}")

    for propagate_type in ["MP", "GNN"]:
        res = GNN_server.train_local_model(
            propagate_type=propagate_type,
            log=False,
            plot=False,
        )
        result[f"server_{propagate_type}"] = res
        bar.set_postfix_str(f"server_{propagate_type}: {res['Test Acc']}")

        res = GNN_server.joint_train_g(
            propagate_type=propagate_type,
            FL=False,
            structure=False,
            log=False,
            plot=False,
        )
        result[f"local_{propagate_type}"] = res
        bar.set_postfix_str(f"local_{propagate_type}: {res['Average']['Test Acc']}")

        res = GNN_server.joint_train_w(
            propagate_type=propagate_type,
            FL=True,
            structure=False,
            log=False,
            plot=False,
        )
        result[f"flwa_{propagate_type}"] = res
        bar.set_postfix_str(f"flwa_{propagate_type}: {res['Average']['Test Acc']}")

        res = GNN_server.joint_train_g(
            propagate_type=propagate_type,
            FL=True,
            structure=False,
            log=False,
            plot=False,
        )
        result[f"flga_{propagate_type}"] = res
        bar.set_postfix_str(f"flga_{propagate_type}: {res['Average']['Test Acc']}")

    propagate_type = "GNN"
    res = FedSage_server.train_fedSage_plus(
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result[f"fedsage+_WA_{propagate_type}"] = res["WA"]
    bar.set_postfix_str(
        f"fedsage+_WA_{propagate_type}: {res['WA']['Average']['Test Acc']}"
    )
    result[f"fedsage+_GA_{propagate_type}"] = res["GA"]
    bar.set_postfix_str(
        f"fedsage+_GA_{propagate_type}: {res['GA']['Average']['Test Acc']}"
    )

    # for structure_type in ["random"]:
    for structure_type in ["degree", "GDV", "node2vec", "random"]:
        for propagate_type in ["MP", "GNN"]:
            res = GNN_server.joint_train_w(
                propagate_type=propagate_type,
                FL=True,
                structure=True,
                structure_type=structure_type,
                log=False,
                plot=False,
            )
            result[f"{structure_type}_sdwa_{propagate_type}"] = res
            bar.set_postfix_str(
                f"{structure_type}_sdwa_{propagate_type}: {res['Average']['Test Acc']}"
            )

            res = GNN_server.joint_train_g(
                propagate_type=propagate_type,
                FL=True,
                structure=True,
                structure_type=structure_type,
                log=False,
                plot=False,
            )
            result[f"{structure_type}_sdga_{propagate_type}"] = res
            bar.set_postfix_str(
                f"{structure_type}_sdga_{propagate_type}: {res['Average']['Test Acc']}"
            )

    return result


def calc_average_results(result, average_result, i):
    if average_result == None:
        return deepcopy(result)

    for key, val in result.items():
        if isinstance(val, float):
            average_result[key] = (average_result[key] * i + val) / (i + 1)
        else:
            average_result[key] = calc_average_results(val, average_result[key], i)

    return average_result


if __name__ == "__main__":
    save_path = f"./results/{config.dataset.dataset_name}/{config.structure_model.structure_type}/average/"
    _LOGGER = get_logger(
        name=f"average_{config.dataset.dataset_name}_{config.structure_model.structure_type}",
        log_on_file=True,
        save_path=save_path,
    )
    _LOGGER2 = get_logger(
        name=f"all_results_{config.dataset.dataset_name}_{config.structure_model.structure_type}",
        terminal=False,
        log_on_file=True,
        save_path=save_path,
    )

    log_config(_LOGGER)

    graph: Graph = define_graph()
    MLP_server = MLPServer(
        graph,
        graph.num_classes,
    )

    GNN_server = GNNServer(
        graph,
        graph.num_classes,
    )

    FedSage_server = FedSAGEServer(
        graph,
        graph.num_classes,
    )

    rep = 10
    average_result = None

    bar = tqdm(total=rep)
    for i in range(rep):
        result = run(
            graph,
            MLP_server,
            GNN_server,
            FedSage_server,
            bar,
        )
        _LOGGER2.info(f"Run id: {i}")
        _LOGGER2.info(json.dumps(result, indent=4))

        average_result = calc_average_results(result, average_result, i)
        save_average_result(average_result)
        bar.update()

    _LOGGER.info(json.dumps(average_result, indent=4))

    save_average_result(average_result)
