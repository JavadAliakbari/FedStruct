import os
from copy import deepcopy
import random

import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.utils.utils import *
from src.GNN_server import GNNServer
from src.MLP_server import MLPServer
from src.fedsage_server import FedSAGEServer
from src.utils.graph import Graph
from src.utils.config_parser import Config
from src.utils.graph_partitioning import (
    create_mend_graph,
    create_mend_graph2,
    partition_graph,
)

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


def create_clients(
    graph: Graph,
    MLP_server: MLPServer,
    GNN_server: GNNServer,
    GNN_server2: GNNServer,
    FedSage_server: FedSAGEServer,
    train_ratio=config.subgraph.train_ratio,
    test_ratio=config.subgraph.test_ratio,
    num_subgraphs=config.subgraph.num_subgraphs,
    partitioning=config.subgraph.partitioning,
):
    graph.add_masks(train_size=train_ratio, test_size=test_ratio)

    MLP_server.remove_clients()
    GNN_server.remove_clients()
    GNN_server2.remove_clients()
    FedSage_server.remove_clients()

    subgraphs = partition_graph(graph, num_subgraphs, partitioning)

    for subgraph in subgraphs:
        MLP_server.add_client(subgraph)
        GNN_server.add_client(subgraph)
        FedSage_server.add_client(subgraph)
        mend_graph = create_mend_graph2(subgraph, graph)
        GNN_server2.add_client(mend_graph)


def get_MLP_results(
    MLP_server: MLPServer,
    bar: tqdm,
    epochs=config.model.epoch_classifier,
):
    result = {}
    MLP_runs = {
        "local_MLP": [MLP_server.joint_train_g, False],
        "flwa_MLP": [MLP_server.joint_train_w, True],
        "flga_MLP": [MLP_server.joint_train_g, True],
    }
    res = MLP_server.train_local_model(epochs=epochs, log=False, plot=False)
    result[f"server_MLP"] = res
    bar.set_postfix_str(f"server_MLP: {res['Test Acc']}")

    for name, run in MLP_runs.items():
        res = run[0](
            epochs=epochs,
            FL=run[1],
            log=False,
            plot=False,
        )
        result[f"{name}"] = res
        bar.set_postfix_str(f"{name}: {res['Average']['Test Acc']}")

    return result


def get_Fedsage_results(
    FedSage_server: FedSAGEServer,
    bar: tqdm,
    epochs=config.model.epoch_classifier,
):
    result = {}
    res = FedSage_server.train_fedSage_plus(
        epochs=epochs,
        propagate_type="GNN",
        log=False,
        plot=False,
    )
    result[f"fedsage+_WA_"] = res["WA"]
    bar.set_postfix_str(f"fedsage+_WA: {res['WA']['Average']['Test Acc']}")
    result[f"fedsage+_GA"] = res["GA"]
    bar.set_postfix_str(f"fedsage+_GA: {res['GA']['Average']['Test Acc']}")

    return result


def get_Fedsage_ideal_reults(
    GNN_server2: GNNServer,
    bar: tqdm,
    epochs=config.model.epoch_classifier,
):
    result = {}

    GNN_runs = {
        "fedsage_ideal_w": [GNN_server2.joint_train_w, True, False, ""],
        "fedsage_ideal_g": [GNN_server2.joint_train_g, True, False, ""],
    }

    for propagate_type in ["DGCN", "GNN"]:
        for name, run in GNN_runs.items():
            res = run[0](
                epochs=epochs,
                propagate_type=propagate_type,
                FL=run[1],
                structure=run[2],
                structure_type=run[3],
                log=False,
                plot=False,
            )
            result[f"{name}_{propagate_type}"] = res
            bar.set_postfix_str(
                f"{name}_{propagate_type}: {res['Average']['Test Acc']}"
            )

    return result


def get_GNN_results(
    GNN_server: GNNServer,
    bar: tqdm,
    epochs=config.model.epoch_classifier,
):
    result = {}

    funcs = {
        "flwa": GNN_server.joint_train_w,
        "flga": GNN_server.joint_train_g,
    }
    GNN_runs = {
        # "local": [GNN_server.joint_train_g, False, False, ""],
    }

    # for method in ["flwa", "flga"]:
    for method in ["flga"]:
        # GNN_runs[f"{method}"] = [funcs[method], True, False, ""]
        for structure_type in ["degree", "GDV", "node2vec", "random"]:
            name = f"{method}_{structure_type}"
            GNN_runs[name] = [funcs[method], True, True, structure_type]

    # for propagate_type in ["DGCN", "GNN"]:
    for propagate_type in ["DGCN"]:
        # res = GNN_server.train_local_model(
        #     epochs=epochs,
        #     propagate_type=propagate_type,
        #     log=False,
        #     plot=False,
        # )
        # result[f"server_{propagate_type}"] = res
        # bar.set_postfix_str(f"server_{propagate_type}: {res['Test Acc']}")

        for name, run in GNN_runs.items():
            res = run[0](
                epochs=epochs,
                propagate_type=propagate_type,
                FL=run[1],
                structure=run[2],
                structure_type=run[3],
                log=False,
                plot=False,
            )
            result[f"{name}_{propagate_type}"] = res
            bar.set_postfix_str(
                f"{name}_{propagate_type}: {res['Average']['Test Acc']}"
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


def calc_average_std_result(results, res_type="Test Acc"):
    results_dict = lod2dol(results)

    average_result = {}
    for method, res in results_dict.items():
        dict_of_clients = lod2dol(res)
        method_results = {}
        for client_id, vals in dict_of_clients.items():
            try:
                final_vals = lod2dol(vals)[res_type]
            except:
                final_vals = vals
            method_results[client_id] = (
                f"{np.mean(final_vals):0.5f}\u00B1{np.std(final_vals):0.5f}"
            )
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
