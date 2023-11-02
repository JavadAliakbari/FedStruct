from copy import deepcopy
import random
import json

import torch
import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.datasets import (
    TUDataset,
    Planetoid,
    HeterophilousGraphDataset,
    WikipediaNetwork,
)
from torch_geometric.utils import to_undirected, remove_self_loops
from tqdm import tqdm

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

config = Config()


def define_graph():
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
        )
    else:
        num_classes = max(graph.y).item() + 1

    graph.add_masks(
        train_size=config.subgraph.train_ratio,
        test_size=config.subgraph.test_ratio,
    )
    graph.num_classes = num_classes

    return graph


def run(
    graph,
    MLP_server: Server,
    degree_server: Server,
    GDV_server: Server,
    node2vec_server: Server,
    random_server: Server,
):
    MLP_server.remove_clients()
    degree_server.remove_clients()
    GDV_server.remove_clients()
    node2vec_server.remove_clients()
    random_server.remove_clients()

    subgraphs = louvain_graph_cut(graph)

    for subgraph in subgraphs:
        MLP_server.add_client(subgraph)
        degree_server.add_client(subgraph)
        GDV_server.add_client(subgraph)
        node2vec_server.add_client(subgraph)
        random_server.add_client(subgraph)

    result = {}

    MLP_server.train_local_classifier(
        config.model.epoch_classifier, log=False, plot=False
    )
    test_acc = MLP_server.test_local_classifier()
    result["server_mlp"] = test_acc

    result["local_mlp"] = MLP_server.train_local_classifiers(
        config.model.epoch_classifier, log=False, plot=False
    )
    result["flwa_mlp"] = MLP_server.train_FLWA(
        config.model.epoch_classifier, log=False, plot=False
    )
    result["flga_mlp"] = MLP_server.train_FLGA(
        config.model.epoch_classifier, log=False, plot=False
    )

    propagate_type = "MP"
    degree_server.train_local_classifier(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    test_acc = degree_server.test_local_classifier()
    result["server_mp"] = test_acc

    result["local_mp"] = degree_server.train_local_classifiers(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["flwa_mp"] = degree_server.train_FLWA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["flga_mp"] = degree_server.train_FLGA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )

    propagate_type = "GNN"
    degree_server.train_local_classifier(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    test_acc = degree_server.test_local_classifier()
    result["server_gnn"] = test_acc

    result["local_gnn"] = degree_server.train_local_classifiers(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["flwa_gnn"] = degree_server.train_FLWA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["flga_gnn"] = degree_server.train_FLGA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )

    propagate_type = "MP"
    config.structure_model.structure_type = "degree"
    degree_server.initialized = False
    config.structure_model.num_structural_features = 256
    result["degree_sd_mp"] = degree_server.train_SD_Server(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["degree_sdwa_mp"] = degree_server.train_SDWA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["degree_sdga_mp"] = degree_server.train_SDGA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )

    propagate_type = "GNN"
    degree_server.initialized = False
    result["degree_sd_gnn"] = degree_server.train_SD_Server(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["degree_sdwa_gnn"] = degree_server.train_SDWA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["degree_sdga_gnn"] = degree_server.train_SDGA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )

    propagate_type = "MP"
    config.structure_model.structure_type = "GDV"
    config.structure_model.num_structural_features = 73
    GDV_server.initialized = False
    result["GDV_sd_mp"] = GDV_server.train_SD_Server(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["GDV_sdwa_mp"] = GDV_server.train_SDWA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["GDV_sdga_mp"] = GDV_server.train_SDGA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )

    propagate_type = "GNN"
    GDV_server.initialized = False
    result["GDV_sd_gnn"] = GDV_server.train_SD_Server(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["GDV_sdwa_gnn"] = GDV_server.train_SDWA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["GDV_sdga_gnn"] = GDV_server.train_SDGA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )

    propagate_type = "MP"
    config.structure_model.structure_type = "node2vec"
    config.structure_model.num_structural_features = 256
    node2vec_server.initialized = False
    result["node2vec_sd_mp"] = node2vec_server.train_SD_Server(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["node2vec_sdwa_mp"] = node2vec_server.train_SDWA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["node2vec_sdga_mp"] = node2vec_server.train_SDGA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )

    propagate_type = "GNN"
    node2vec_server.initialized = False
    result["node2vec_sd_gnn"] = node2vec_server.train_SD_Server(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["node2vec_sdwa_gnn"] = node2vec_server.train_SDWA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["node2vec_dga_gnn"] = node2vec_server.train_SDGA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )

    propagate_type = "MP"
    config.structure_model.structure_type = "random"
    config.structure_model.num_structural_features = 256
    random_server.initialized = False
    result["random_sd_mp"] = random_server.train_SD_Server(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["random_sdwa_mp"] = random_server.train_SDWA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["random_sdga_mp"] = random_server.train_SDGA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )

    propagate_type = "GNN"
    random_server.initialized = False
    result["random_sd_gnn"] = random_server.train_SD_Server(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["random_sdwa_gnn"] = random_server.train_SDWA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
    )
    result["random_sdga_gnn"] = random_server.train_SDGA(
        config.model.epoch_classifier,
        propagate_type=propagate_type,
        log=False,
        plot=False,
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

    graph = define_graph()
    MLP_server = Server(
        graph,
        graph.num_classes,
        classifier_type="MLP",
    )

    config.structure_model.structure_type = "degree"
    config.structure_model.num_structural_features = 256
    degree_server = Server(
        graph,
        graph.num_classes,
        classifier_type="GNN",
    )

    config.structure_model.structure_type = "GDV"
    config.structure_model.num_structural_features = 73
    GDV_server = Server(
        graph,
        graph.num_classes,
        classifier_type="GNN",
    )

    config.structure_model.structure_type = "node2vec"
    config.structure_model.num_structural_features = 256
    node2vec_server = Server(
        graph,
        graph.num_classes,
        classifier_type="GNN",
    )

    config.structure_model.structure_type = "random"
    config.structure_model.num_structural_features = 256
    random_server = Server(
        graph,
        graph.num_classes,
        classifier_type="GNN",
    )

    rep = 10
    average_result = None

    bar = tqdm(total=rep)
    for i in range(rep):
        result = run(
            graph,
            MLP_server,
            degree_server,
            GDV_server,
            node2vec_server,
            random_server,
        )
        _LOGGER2.info(f"Run id: {i}")
        _LOGGER2.info(json.dumps(result, indent=4))

        average_result = calc_average_results(result, average_result, i)
        # bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
        metrics = {
            # "local_mlp": average_result["server_mlp"]
            # "local_mlp": average_result["server_mlp"]
            # "server": average_result["server_gnn"],
            # "local": average_result["local_gnn"]["Average"],
            # "flwa": average_result["flwa_gnn"]["Average"],
            # "flga": average_result["flga_gnn"]["Average"],
            # "sd": average_result["sd_gnn"]["Server"],
            # "sdwa": average_result["sdwa_gnn"]["Average"],
            # "sdga": average_result["sdga_mp"]["Average"],
            # "server": average_result["server_mp"],
            # "local": average_result["local_mp"]["Average"],
            # "flwa": average_result["flwa_mp"]["Average"],
            # "flga": average_result["flga_mp"]["Average"],
            # "sd": average_result["sd_mp"]["Server"],
            # "sdwa": average_result["sdwa_mp"]["Average"],
            # "sdga": average_result["sdga_mp"]["Average"],
        }
        # bar.set_postfix(metrics)
        bar.update()

    _LOGGER.info(json.dumps(average_result, indent=4))
