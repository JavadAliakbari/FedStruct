import os
import sys
import json
from copy import deepcopy

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

from tqdm import tqdm

from src import *
from src.utils.define_graph import define_graph
from src.GNN.GNN_server import GNNServer
from src.utils.logger import getLOGGER
from src.utils.graph_partitioning import (
    partition_graph,
)
from src.simulations.simulation_utils import (
    calc_average_std_result,
    save_average_result,
)
from src.utils.graph import Graph


def create_clients(
    graph: Graph,
    GNN_server: GNNServer,
    train_ratio=config.subgraph.train_ratio,
    test_ratio=config.subgraph.test_ratio,
    num_subgraphs=config.subgraph.num_subgraphs,
    partitioning=config.subgraph.partitioning,
):
    graph.add_masks(train_size=train_ratio, test_size=test_ratio)

    GNN_server.remove_clients()

    subgraphs = partition_graph(graph, num_subgraphs, partitioning)

    for subgraph in subgraphs:
        GNN_server.add_client(subgraph)


def get_GNN_results(
    GNN_server: GNNServer,
    bar: tqdm,
    epochs=config.model.iterations,
    smodel_types=["GNN", "DGCN"],
):
    funcs = {
        "server": GNN_server.train_local_model,
        "flwa": GNN_server.joint_train_w,
        "flga": GNN_server.joint_train_g,
    }

    # for method in ["flwa", "flga"]:
    GNN_runs = {}
    GNN_runs[f"server_feature_GNN"] = [
        funcs["server"],
        "GNN",
        False,
        "feature",
        "",
    ]
    GNN_runs[f"flga_feature_GNN"] = [
        funcs["flga"],
        "GNN",
        True,
        "feature",
        "",
    ]
    GNN_runs[f"server_structure_node2vec_GNN"] = [
        funcs["server"],
        "GNN",
        False,
        "structure",
        "node2vec",
    ]
    GNN_runs[f"flga_f+s_node2vec_DGCN"] = [
        funcs["flga"],
        "DGCN",
        True,
        "f+s",
        "node2vec",
    ]

    result = {}
    for name, run in GNN_runs.items():
        res = run[0](
            epochs=epochs,
            smodel_type=run[1],
            FL=run[2],
            data_type=run[3],
            structure_type=run[4],
            log=False,
            plot=False,
        )
        result[name] = res
        try:
            bar.set_postfix_str(f"{name}: {res['Average']['Test Acc']}")
        except:
            bar.set_postfix_str(f"{name}: {res['Test Acc']}")

    return result


if __name__ == "__main__":
    graph = define_graph(config.dataset.dataset_name)
    true_abar = calc_a(
        graph.edge_index,
        graph.num_nodes,
        config.structure_model.DGCN_layers,
        pruning=False,
    )

    rep = 10

    # for partitioning in [config.subgraph.partitioning]:
    for partitioning in ["random", "louvain", "kmeans"]:
        # for num_subgraphs in [config.subgraph.num_subgraphs]:
        for num_subgraphs in [10]:
            # for num_subgraphs in [5, 10, 20]:
            for train_ratio in [config.subgraph.train_ratio]:
                # for train_ratio in np.arange(0.1, 0.65, 0.05):
                test_ratio = config.subgraph.test_ratio
                # test_ratio = (1 - train_ratio) / 2
                epochs = config.model.iterations
                # epochs = int(train_ratio * 100 + 30)

                save_path = (
                    "./results/Neurips/structure/"
                    f"{config.dataset.dataset_name}/"
                    f"{partitioning}/"
                    f"{num_subgraphs}/"
                    f"{train_ratio}/"
                )

                os.makedirs(save_path, exist_ok=True)

                LOGGER = getLOGGER(
                    name=f"average_{now}_{config.dataset.dataset_name}",
                    log_on_file=True,
                    save_path=save_path,
                )
                LOGGER2 = getLOGGER(
                    name=f"all_{now}_{config.dataset.dataset_name}",
                    terminal=False,
                    log_on_file=True,
                    save_path=save_path,
                )
                LOGGER.info(json.dumps(config.config, indent=4))

                bar = tqdm(total=rep)
                results = []
                for i in range(rep):
                    graph.abar = deepcopy(true_abar)

                    GNN_server = GNNServer(graph)
                    create_clients(
                        graph,
                        GNN_server,
                        train_ratio=train_ratio,
                        test_ratio=test_ratio,
                        num_subgraphs=num_subgraphs,
                        partitioning=partitioning,
                    )
                    model_results = {}

                    GNN_result_true = get_GNN_results(
                        GNN_server,
                        bar=bar,
                        epochs=epochs,
                    )

                    model_results.update(GNN_result_true)

                    LOGGER2.info(f"Run id: {i}")
                    LOGGER2.info(json.dumps(model_results, indent=4))

                    results.append(model_results)

                    average_result = calc_average_std_result(results)
                    file_name = f"{save_path}{now}_{config.dataset.dataset_name}.csv"
                    save_average_result(average_result, file_name)

                    bar.update()

                LOGGER.info(json.dumps(average_result, indent=4))

                LOGGER.handlers.clear()
                LOGGER2.handlers.clear()
