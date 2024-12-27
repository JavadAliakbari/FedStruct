import numpy as np
import pandas as pd
from tqdm import tqdm

from src import *
from src.GNN.GNN_server import GNNServer
from src.MLP.MLP_server import MLPServer
from src.FedPub.fedpub_server import FedPubServer
from src.FedGCN.FedGCN_server import FedGCNServer
from src.fedsage.fedsage_server import FedSAGEServer
from src.utils.graph import Graph
from src.utils.graph_partitioning import (
    create_mend_graph,
    fedGCN_partitioning,
    partition_graph,
)


def create_clients(
    graph: Graph,
    MLP_server: MLPServer,
    GNN_server: GNNServer,
    GNN_server_ideal: GNNServer,
    FedSage_server: FedSAGEServer,
    FedPub_server: FedPubServer,
    Fedgcn_server1: FedGCNServer,
    Fedgcn_server2: FedGCNServer,
    train_ratio=config.subgraph.train_ratio,
    test_ratio=config.subgraph.test_ratio,
    num_subgraphs=config.subgraph.num_subgraphs,
    partitioning=config.subgraph.partitioning,
):
    graph.add_masks(train_ratio=train_ratio, test_ratio=test_ratio)

    MLP_server.reset_clients()
    GNN_server.reset_clients()
    GNN_server_ideal.reset_clients()
    FedSage_server.reset_clients()
    FedPub_server.reset_clients()
    Fedgcn_server1.reset_clients()
    Fedgcn_server2.reset_clients()

    subgraphs = partition_graph(graph, num_subgraphs, partitioning)

    for subgraph in subgraphs:
        MLP_server.add_client(subgraph)
        GNN_server.add_client(subgraph)
        FedSage_server.add_client(subgraph)
        FedPub_server.add_client(subgraph)
        mend_graph = create_mend_graph(subgraph, graph)
        GNN_server_ideal.add_client(mend_graph)

    fedgcn_subgraphs1 = fedGCN_partitioning(
        graph, config.subgraph.num_subgraphs, method=partitioning, num_hops=1
    )
    for subgraph in fedgcn_subgraphs1:
        Fedgcn_server1.add_client(subgraph)

    fedgcn_subgraphs2 = fedGCN_partitioning(
        graph, config.subgraph.num_subgraphs, method=partitioning, num_hops=2
    )
    for subgraph in fedgcn_subgraphs2:
        Fedgcn_server2.add_client(subgraph)


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
    epochs=config.fedpub.epochs,
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


def get_Fedgcn_results(
    FedGCN_server: FedGCNServer,
    bar: tqdm,
    epochs=config.model.iterations,
    num_hops=2,
):
    result = {}
    res = FedGCN_server.joint_train_w(
        epochs=epochs,
        log=False,
        plot=False,
    )
    result[f"fedgcn{num_hops}"] = res
    bar.set_postfix_str(f"fedgcn{num_hops}: {res['Average']['Test Acc']}")

    return result


def get_Fedsage_ideal_reults(
    GNN_server2: GNNServer,
    bar: tqdm,
    epochs=config.model.iterations,
    fmodel_types=["DGCN", "GNN"],
):
    result = {}

    GNN_runs = {
        # "fedsage_ideal_w": [GNN_server2.joint_train_w, True, False, ""],
        "fedsage_ideal_g": [GNN_server2.joint_train_g, True, "feature"],
    }

    for fmodel_type in fmodel_types:
        for name, run in GNN_runs.items():
            res = run[0](
                epochs=epochs,
                fmodel_type=fmodel_type,
                FL=run[1],
                data_type=run[2],
                log=False,
                plot=False,
            )
            result[f"{name}_{fmodel_type}"] = res
            bar.set_postfix_str(f"{name}_{fmodel_type}: {res['Average']['Test Acc']}")

    return result


def get_GNN_results(
    GNN_server: GNNServer,
    bar: tqdm,
    epochs=config.model.iterations,
    smodel_types=["DGCN", "GNN"],
    structure_types=["degree", "fedstar", "GDV", "node2vec", "hop2vec"],
):
    result = {}

    funcs = {
        "flwa": GNN_server.joint_train_w,
        "flga": GNN_server.joint_train_g,
    }
    GNN_runs = {
        "local": [GNN_server.joint_train_g, False, "feature", ""],
    }
    GNN_runs[f"flwa"] = [funcs["flwa"], True, "feature", ""]

    # for method in ["flwa", "flga"]:
    for method in ["flga"]:
        GNN_runs[f"{method}"] = [funcs[method], True, "feature", ""]
        for structure_type in structure_types:
            name = f"{method}_{structure_type}"
            GNN_runs[name] = [funcs[method], True, "f+s", structure_type]

    for smodel_type in smodel_types:
        res = GNN_server.train_local_model(
            epochs=epochs,
            smodel_type=smodel_type,
            fmodel_type=smodel_type,
            log=False,
            plot=False,
        )
        result[f"server_{smodel_type}"] = res
        bar.set_postfix_str(f"server_{smodel_type}: {res['Test Acc']}")

        for name, run in GNN_runs.items():
            res = run[0](
                epochs=epochs,
                smodel_type=smodel_type,
                fmodel_type=smodel_type,
                FL=run[1],
                data_type=run[2],
                structure_type=run[3],
                log=False,
                plot=False,
            )
            result[f"{name}_{smodel_type}"] = res
            bar.set_postfix_str(f"{name}_{smodel_type}: {res['Average']['Test Acc']}")

    return result


def calc_average_std_result(results, res_type="Test Acc"):
    results_dict = lod2dol(results)

    average_result = {}
    for method, res in results_dict.items():
        dict_of_clients = lod2dol(res)
        method_results = {}
        for client_id, vals in dict_of_clients.items():
            try:
                temp = lod2dol(vals)
                if not res_type in temp.keys():
                    break
                final_vals = temp[res_type]
            except:
                final_vals = vals
            method_results[client_id] = (
                f"{np.mean(final_vals):0.5f}\u00B1{np.std(final_vals):0.5f}"
            )
            # method_results[client_id] = [np.mean(final_vals), np.std(final_vals)]
        if len(method_results) > 0:
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
