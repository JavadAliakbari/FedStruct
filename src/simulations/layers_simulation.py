import os
import json
import sys

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

from tqdm import tqdm

from src import *
from src.utils.graph import Graph
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


def run(
    graph: Graph,
    GNN_server: GNNServer,
    bar: tqdm,
    epochs=config.model.iterations,
    train_ratio=config.subgraph.train_ratio,
    test_ratio=config.subgraph.test_ratio,
    num_subgraphs=config.subgraph.num_subgraphs,
    partitioning=config.subgraph.partitioning,
):
    graph.add_masks(train_ratio=train_ratio, test_ratio=test_ratio)

    GNN_server.reset_clients()

    subgraphs = partition_graph(graph, num_subgraphs, partitioning)

    for subgraph in subgraphs:
        GNN_server.add_client(subgraph)

    result = {}
    res = GNN_server.joint_train_g(
        epochs=epochs,
        data_type="feature",
        fmodel_type="GNN",
        FL=True,
        log=False,
        plot=False,
    )
    result["flga_GNN"] = res
    bar.set_postfix_str(f"flga_GNN: {res['Average']['Test Acc']}")

    # for structure_type in ["hop2vec"]:
    for structure_type in ["degree", "fedstar", "node2vec", "hop2vec"]:
        for smodel_type in ["DGCN"]:
            for data_type in ["f+s", "structure"]:
                res = GNN_server.joint_train_g(
                    epochs=epochs,
                    data_type=data_type,
                    smodel_type=smodel_type,
                    fmodel_type=smodel_type,
                    structure_type=structure_type,
                    FL=True,
                    log=False,
                    plot=False,
                )
                result[f"flga_{structure_type}_{smodel_type}_{data_type}"] = res
                bar.set_postfix_str(
                    f"flga_{structure_type}_{smodel_type}_{data_type}: {res['Average']['Test Acc']}"
                )

    return result


if __name__ == "__main__":
    graph = define_graph(config.dataset.dataset_name)

    GNN_server = GNNServer(graph)

    rep = 50

    # for partitioning in [config.subgraph.partitioning]:
    for DGCN_layers in range(0, 51):
        abar = calc_abar(graph.edge_index, graph.num_nodes, DGCN_layers, pruning=False)
        graph.abar = abar
        config.feature_model.gnn_layer_sizes = [64] * DGCN_layers
        for partitioning in ["kmeans"]:
            # for num_subgraphs in [config.subgraph.num_subgraphs]:
            for num_subgraphs in [5]:
                for train_ratio in [config.subgraph.train_ratio]:
                    # for train_ratio in np.arange(0.1, 0.65, 0.05):
                    test_ratio = config.subgraph.test_ratio
                    # test_ratio = (1 - train_ratio) / 2
                    epochs = config.model.iterations
                    # epochs = int(train_ratio * 100 + 30)

                    save_path = (
                        "./results/layers/no_pruning/"
                        f"{config.dataset.dataset_name}/"
                        f"{partitioning}/"
                        f"{num_subgraphs}/"
                        f"{train_ratio}/"
                        f"{DGCN_layers}/"
                    )
                    os.makedirs(save_path, exist_ok=True)

                    LOGGER = getLOGGER(
                        name=f"average_{config.dataset.dataset_name}_{train_ratio}",
                        log_on_file=True,
                        save_path=save_path,
                    )
                    LOGGER2 = getLOGGER(
                        name=f"all_{config.dataset.dataset_name}_{train_ratio}",
                        terminal=False,
                        log_on_file=True,
                        save_path=save_path,
                    )
                    LOGGER.info(json.dumps(config.config, indent=4))

                    bar = tqdm(total=rep)
                    results = []
                    for i in range(rep):
                        result = run(
                            graph,
                            GNN_server,
                            bar=bar,
                            epochs=epochs,
                            train_ratio=train_ratio,
                            test_ratio=test_ratio,
                            num_subgraphs=num_subgraphs,
                            partitioning=partitioning,
                        )
                        LOGGER2.info(f"Run id: {i}")
                        LOGGER2.info(json.dumps(result, indent=4))

                        results.append(result)

                        average_result = calc_average_std_result(results, "Test Acc")
                        file_name = f"{save_path}final_result_{train_ratio}.csv"
                        save_average_result(average_result, file_name)
                        average_result_f = calc_average_std_result(
                            results, "Test Acc F"
                        )
                        file_name_f = f"{save_path}final_result_F_{train_ratio}.csv"
                        save_average_result(average_result_f, file_name_f)
                        average_result_s = calc_average_std_result(
                            results, "Test Acc S"
                        )
                        file_name_s = f"{save_path}final_result_S_{train_ratio}.csv"
                        save_average_result(average_result_s, file_name_s)

                        bar.update()
                    LOGGER.info("Test")
                    LOGGER.info(json.dumps(average_result, indent=4))
                    LOGGER.info("Test F")
                    LOGGER.info(json.dumps(average_result_f, indent=4))
                    LOGGER.info("Test S")
                    LOGGER.info(json.dumps(average_result_s, indent=4))

                    LOGGER.handlers.clear()
                    LOGGER2.handlers.clear()
