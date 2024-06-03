import os
import sys
import json

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

from tqdm import tqdm

from src import *
from src.simulations.simulation_utils import *
from src.utils.define_graph import define_graph
from src.utils.logger import getLOGGER
from src.GNN.GNN_server import GNNServer
from src.MLP.MLP_server import MLPServer
from src.FedPub.fedpub_server import FedPubServer
from src.fedsage.fedsage_server import FedSAGEServer

if __name__ == "__main__":
    graph = define_graph(config.dataset.dataset_name)
    true_abar = calc_a(
        graph.edge_index,
        graph.num_nodes,
        config.structure_model.DGCN_layers,
        pruning=False,
    )
    prune_abar = calc_a(
        graph.edge_index,
        graph.num_nodes,
        config.structure_model.DGCN_layers,
        pruning=True,
    )

    MLP_server = MLPServer(graph)

    GNN_server = GNNServer(graph)

    FedSage_server = FedSAGEServer(graph)

    FedPub_server = FedPubServer(graph)

    GNN_server_ideal = GNNServer(graph)

    rep = 10

    for partitioning in ["random", "louvain", "kmeans"]:
        if partitioning == "random":
            num_subgraphs_list = [5, 10, 20]
        else:
            num_subgraphs_list = [10]

        for num_subgraphs in num_subgraphs_list:
            for train_ratio in [config.subgraph.train_ratio]:
                test_ratio = config.subgraph.test_ratio
                epochs = config.model.iterations

                simulation_path = (
                    "./results/Simulation/"
                    f"{config.dataset.dataset_name}/"
                    f"{partitioning}/"
                    f"{num_subgraphs}/"
                    f"{train_ratio}/"
                )

                os.makedirs(simulation_path, exist_ok=True)

                LOGGER = getLOGGER(
                    name=f"average_{now}_{config.dataset.dataset_name}",
                    log_on_file=True,
                    save_path=simulation_path,
                )
                LOGGER2 = getLOGGER(
                    name=f"all_{now}_{config.dataset.dataset_name}",
                    terminal=False,
                    log_on_file=True,
                    save_path=simulation_path,
                )
                LOGGER.info(json.dumps(config.config, indent=4))

                bar = tqdm(total=rep)
                results = []
                for i in range(rep):
                    create_clients(
                        graph,
                        MLP_server,
                        GNN_server,
                        GNN_server_ideal,
                        FedSage_server,
                        FedPub_server,
                        train_ratio=train_ratio,
                        test_ratio=test_ratio,
                        num_subgraphs=num_subgraphs,
                        partitioning=partitioning,
                    )
                    model_results = {}

                    MLP_results = get_MLP_results(
                        MLP_server,
                        bar=bar,
                        epochs=epochs,
                    )
                    model_results.update(MLP_results)
                    Fedsage_results = get_Fedsage_results(
                        FedSage_server,
                        bar=bar,
                        epochs=epochs,
                    )
                    model_results.update(Fedsage_results)
                    Fedpub_results = get_Fedpub_results(
                        FedPub_server,
                        bar=bar,
                        epochs=epochs,
                    )
                    model_results.update(Fedpub_results)
                    Fedsage_ideal_results = get_Fedsage_ideal_reults(
                        GNN_server_ideal,
                        bar=bar,
                        epochs=epochs,
                    )
                    model_results.update(Fedsage_ideal_results)

                    graph.abar = true_abar
                    GNN_result_true = get_GNN_results(
                        GNN_server,
                        bar=bar,
                        epochs=epochs,
                    )

                    GNN_result_true2 = {}
                    for key, val in GNN_result_true.items():
                        GNN_result_true2[f"{key}_true"] = val
                    model_results.update(GNN_result_true2)

                    graph.abar = prune_abar
                    GNN_result_prune = get_GNN_results(
                        GNN_server,
                        bar=bar,
                        epochs=epochs,
                        propagate_types=["DGCN"],
                    )
                    GNN_result_prune2 = {}
                    for key, val in GNN_result_prune.items():
                        GNN_result_prune2[f"{key}_prune"] = val
                    model_results.update(GNN_result_prune2)

                    LOGGER2.info(f"Run id: {i}")
                    LOGGER2.info(json.dumps(model_results, indent=4))

                    results.append(model_results)

                    average_result = calc_average_std_result(results)
                    file_name = (
                        f"{simulation_path}{now}_{config.dataset.dataset_name}.csv"
                    )
                    save_average_result(average_result, file_name)

                    bar.update()

                LOGGER.info(json.dumps(average_result, indent=4))

                LOGGER.handlers.clear()
                LOGGER2.handlers.clear()
