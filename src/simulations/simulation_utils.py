import numpy as np
import pandas as pd
from tqdm import tqdm

from src import *
from src.GNN.GNN_server import GNNServer
from src.MLP.MLP_server import MLPServer
from src.FedPub.fedpub_server import FedPubServer
from src.fedsage.fedsage_server import FedSAGEServer
from src.utils.graph import Graph
from src.utils.graph_partitioning import (
    create_mend_graph,
    partition_graph,
)


def create_clients(
    graph: Graph,
    MLP_server: MLPServer,
    GNN_server: GNNServer,
    GNN_server_ideal: GNNServer,
    FedSage_server: FedSAGEServer,
    FedPub_server: FedPubServer,
    train_ratio=config.subgraph.train_ratio,
    test_ratio=config.subgraph.test_ratio,
    num_subgraphs=config.subgraph.num_subgraphs,
    partitioning=config.subgraph.partitioning,
):
    graph.add_masks(train_size=train_ratio, test_size=test_ratio)

    MLP_server.remove_clients()
    GNN_server.remove_clients()
    GNN_server_ideal.remove_clients()
    FedSage_server.remove_clients()
    FedPub_server.remove_clients()

    subgraphs = partition_graph(graph, num_subgraphs, partitioning)

    for subgraph in subgraphs:
        MLP_server.add_client(subgraph)
        GNN_server.add_client(subgraph)
        FedSage_server.add_client(subgraph)
        FedPub_server.add_client(subgraph)
        mend_graph = create_mend_graph(subgraph, graph)
        GNN_server_ideal.add_client(mend_graph)


def get_MLP_results(
    MLP_server: MLPServer,
    bar: tqdm,
    epochs=config.model.iterations,
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
    epochs=config.model.iterations,
):
    result = {}
    res = FedSage_server.train_fedSage_plus(
        epochs=epochs,
        # propagate_type="GNN",
        model="both",
        log=False,
        plot=False,
    )
    result[f"fedsage+_WA_"] = res["WA"]
    bar.set_postfix_str(f"fedsage+_WA: {res['WA']['Average']['Test Acc']}")
    result[f"fedsage+_GA"] = res["GA"]
    bar.set_postfix_str(f"fedsage+_GA: {res['GA']['Average']['Test Acc']}")

    return result


def get_Fedpub_results(
    FedPub_server: FedPubServer,
    bar: tqdm,
    epochs=config.model.iterations,
):
    result = {}
    res = FedPub_server.start(
        iterations=epochs,
        log=False,
        plot=False,
    )
    result[f"fedpub"] = res
    bar.set_postfix_str(f"fedpub: {res['Average']['Test Acc']}")

    return result


def get_Fedsage_ideal_reults(
    GNN_server2: GNNServer,
    bar: tqdm,
    epochs=config.model.iterations,
):
    result = {}

    GNN_runs = {
        # "fedsage_ideal_w": [GNN_server2.joint_train_w, True, False, ""],
        "fedsage_ideal_g": [GNN_server2.joint_train_g, True, "feature", ""],
    }

    for propagate_type in ["DGCN", "GNN"]:
        for name, run in GNN_runs.items():
            res = run[0](
                epochs=epochs,
                propagate_type=propagate_type,
                FL=run[1],
                data_type=run[2],
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
    epochs=config.model.iterations,
    propagate_types=["DGCN", "GNN"],
):
    result = {}

    funcs = {
        "flwa": GNN_server.joint_train_w,
        "flga": GNN_server.joint_train_g,
    }
    GNN_runs = {
        "local": [GNN_server.joint_train_g, False, "feature", ""],
    }

    # for method in ["flwa", "flga"]:
    for method in ["flga"]:
        GNN_runs[f"{method}"] = [funcs[method], True, "feature", ""]
        for structure_type in ["degree", "fedstar", "GDV", "node2vec", "hop2vec"]:
            name = f"{method}_{structure_type}"
            GNN_runs[name] = [funcs[method], True, "f+s", structure_type]

    for propagate_type in propagate_types:
        # for propagate_type in ["DGCN"]:
        res = GNN_server.train_local_model(
            epochs=epochs,
            propagate_type=propagate_type,
            log=False,
            plot=False,
        )
        result[f"server_{propagate_type}"] = res
        bar.set_postfix_str(f"server_{propagate_type}: {res['Test Acc']}")

        for name, run in GNN_runs.items():
            res = run[0](
                epochs=epochs,
                propagate_type=propagate_type,
                FL=run[1],
                data_type=run[2],
                structure_type=run[3],
                log=False,
                plot=False,
            )
            result[f"{name}_{propagate_type}"] = res
            bar.set_postfix_str(
                f"{name}_{propagate_type}: {res['Average']['Test Acc']}"
            )

    return result


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
