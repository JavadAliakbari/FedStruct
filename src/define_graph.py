import os

import torch
from torch_geometric.datasets import (
    Planetoid,
    HeterophilousGraphDataset,
    WikipediaNetwork,
    Amazon,
)
from torch_geometric.utils import to_undirected, remove_self_loops

from src.utils.graph import Graph
from src.utils.config_parser import Config
from src.utils.create_graph import create_homophilic_graph2, create_heterophilic_graph2

path = os.environ.get("CONFIG_PATH")
config = Config(path)


def define_graph(dataset_name=config.dataset.dataset_name) -> [Graph, int]:
    try:
        dataset = None
        if dataset_name in ["Cora", "PubMed", "CiteSeer"]:
            dataset = Planetoid(
                root=f"/tmp/{dataset_name}",
                name=dataset_name,
            )
            node_ids = torch.arange(dataset[0].num_nodes)
            edge_index = dataset[0].edge_index
            num_classes = dataset.num_classes
        elif dataset_name in ["chameleon", "crocodile", "squirrel"]:
            dataset = WikipediaNetwork(
                root=f"/tmp/{dataset_name}",
                geom_gcn_preprocess=True,
                name=dataset_name,
            )
            node_ids = torch.arange(dataset[0].num_nodes)
            edge_index = dataset[0].edge_index
            num_classes = dataset.num_classes
        elif dataset_name in [
            "Roman-empire",
            "Amazon-ratings",
            "Minesweeper",
            "Tolokers",
            "Questions",
        ]:
            dataset = HeterophilousGraphDataset(
                root=f"/tmp/{dataset_name}",
                name=dataset_name,
            )
        elif config.dataset.dataset_name in ["Computers", "Photo"]:
            dataset = Amazon(
                root=f"/tmp/{config.dataset.dataset_name}",
                name=config.dataset.dataset_name,
            )
        elif dataset_name == "Heterophilic_example":
            num_patterns = 500
            graph = create_heterophilic_graph2(num_patterns, use_random_features=True)
        elif dataset_name == "Homophilic_example":
            num_patterns = 100
            graph = create_homophilic_graph2(num_patterns, use_random_features=True)

    except:
        # _LOGGER.info("dataset name does not exist!")
        return None, 0

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
            dataset_name=config.dataset.dataset_name,
        )
    else:
        num_classes = max(graph.y).item() + 1

    return graph, num_classes
