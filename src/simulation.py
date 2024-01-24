import os
from copy import deepcopy
import random
import json

import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.utils.utils import *
from src.define_graph import define_graph
from src.GNN_server import GNNServer
from src.MLP_server import MLPServer
from src.fedsage_server import FedSAGEServer
from src.utils.graph import Graph
from src.utils.logger import get_logger
from src.utils.config_parser import Config
from src.utils.graph_partitioning import partition_graph
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


def run(
    graph: Graph,
    MLP_server: MLPServer,
    GNN_server: GNNServer,
    FedSage_server: FedSAGEServer,
    bar: tqdm,
    epochs=config.model.epoch_classifier,
    train_ratio=config.subgraph.train_ratio,
    test_ratio=config.subgraph.test_ratio,
    num_subgraphs=config.subgraph.num_subgraphs,
    partitioning=config.subgraph.partitioning,
):
    graph.add_masks(train_size=train_ratio, test_size=test_ratio)

    MLP_server.remove_clients()
    GNN_server.remove_clients()
    FedSage_server.remove_clients()

    subgraphs = partition_graph(graph, num_subgraphs, partitioning)

    for subgraph in subgraphs:
        MLP_server.add_client(subgraph)
        GNN_server.add_client(subgraph)
        FedSage_server.add_client(subgraph)

    result = {}

    res = MLP_server.train_local_model(epochs=epochs, log=False, plot=False)
    result["server_mlp"] = res
    bar.set_postfix_str(f"server_mlp: {res['Test Acc']}")
    res = MLP_server.joint_train_g(epochs=epochs, FL=False, log=False, plot=False)
    result["local_mlp"] = res
    bar.set_postfix_str(f"local_mlp: {res['Average']['Test Acc']}")
    # res = MLP_server.joint_train_w(epochs=epochs, FL=True, log=False, plot=False)
    # result["flwa_mlp"] = res
    # bar.set_postfix_str(f"flwa_mlp: {res['Average']['Test Acc']}")
    res = MLP_server.joint_train_g(epochs=epochs, FL=True, log=False, plot=False)
    result["flga_mlp"] = res
    bar.set_postfix_str(f"flga_mlp: {res['Average']['Test Acc']}")

    for propagate_type in ["DGCN", "GNN"]:
        res = GNN_server.train_local_model(
            epochs=epochs,
            propagate_type=propagate_type,
            log=False,
            plot=False,
        )
        result[f"server_{propagate_type}"] = res
        bar.set_postfix_str(f"server_{propagate_type}: {res['Test Acc']}")

        res = GNN_server.joint_train_g(
            epochs=epochs,
            propagate_type=propagate_type,
            FL=False,
            structure=False,
            log=False,
            plot=False,
        )
        result[f"local_{propagate_type}"] = res
        bar.set_postfix_str(f"local_{propagate_type}: {res['Average']['Test Acc']}")

        # res = GNN_server.joint_train_w(
        #     epochs=epochs,
        #     propagate_type=propagate_type,
        #     FL=True,
        #     structure=False,
        #     log=False,
        #     plot=False,
        # )
        # result[f"flwa_{propagate_type}"] = res
        # bar.set_postfix_str(f"flwa_{propagate_type}: {res['Average']['Test Acc']}")

        res = GNN_server.joint_train_g(
            epochs=epochs,
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
        epochs=epochs,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    # result[f"fedsage+_WA_{propagate_type}"] = res["WA"]
    # bar.set_postfix_str(
    #     f"fedsage+_WA_{propagate_type}: {res['WA']['Average']['Test Acc']}"
    # )
    result[f"fedsage+_GA_{propagate_type}"] = res["GA"]
    bar.set_postfix_str(
        f"fedsage+_GA_{propagate_type}: {res['GA']['Average']['Test Acc']}"
    )

    # for structure_type in ["random"]:
    for structure_type in ["degree", "GDV", "node2vec", "random"]:
        for propagate_type in ["DGCN", "GNN"]:
            # res = GNN_server.joint_train_w(
            #     epochs=epochs,
            #     propagate_type=propagate_type,
            #     FL=True,
            #     structure=True,
            #     structure_type=structure_type,
            #     log=False,
            #     plot=False,
            # )
            # result[f"{structure_type}_sdwa_{propagate_type}"] = res
            # bar.set_postfix_str(
            #     f"{structure_type}_sdwa_{propagate_type}: {res['Average']['Test Acc']}"
            # )

            res = GNN_server.joint_train_g(
                epochs=epochs,
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


def calc_average_std_result(results):
    results_dict = lod2dol(results)

    average_result = {}
    for method, res in results_dict.items():
        dict_of_clients = lod2dol(res)
        method_results = {}
        for client_id, vals in dict_of_clients.items():
            try:
                final_vals = lod2dol(vals)["Test Acc"]
            except:
                final_vals = vals
            method_results[
                client_id
            ] = f"{np.mean(final_vals):0.5f}\u00B1{np.std(final_vals):0.5f}"
            # method_results[client_id] = [np.mean(final_vals), np.std(final_vals)]

        average_result[method] = method_results

    return average_result


def save_average_result(average_result, file_name="results.csv", save_path="./"):
    final_result = {}
    for key, val in average_result.items():
        if "Average" in val.keys():
            final_result[key] = val["Average"]
        else:
            final_result[key] = val["Test Acc"]

    df = pd.DataFrame.from_dict(final_result, orient="index")
    df.to_csv(f"{save_path}{file_name}")


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

    graph, num_classes = define_graph()
    graph.add_masks(
        train_size=config.subgraph.train_ratio,
        test_size=config.subgraph.test_ratio,
    )

    MLP_server = MLPServer(
        graph,
        num_classes,
    )

    GNN_server = GNNServer(
        graph,
        num_classes,
    )

    FedSage_server = FedSAGEServer(
        graph,
        num_classes,
    )

    rep = 10
    average_result = None

    bar = tqdm(total=rep)
    results = []
    for i in range(rep):
        result = run(
            graph,
            MLP_server,
            GNN_server,
            FedSage_server,
            bar,
            epochs=config.model.epoch_classifier,
            train_ratio=config.subgraph.train_ratio,
            test_ratio=config.subgraph.test_ratio,
            num_subgraphs=config.subgraph.num_subgraphs,
            partitioning=config.subgraph.partitioning,
        )
        _LOGGER2.info(f"Run id: {i}")
        _LOGGER2.info(json.dumps(result, indent=4))

        results.append(result)

        average_result = calc_average_std_result(results)

        file_name = f"{save_path}final_result.csv"
        save_average_result(average_result, file_name)
        bar.update()

    _LOGGER.info(json.dumps(average_result, indent=4))
