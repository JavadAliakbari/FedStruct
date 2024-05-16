import os

import torch
from torch_geometric.datasets import (
    Planetoid,
    HeterophilousGraphDataset,
    WikipediaNetwork,
    Amazon,
    Actor,
)
from torch_geometric.utils import to_undirected, remove_self_loops

from src.utils.graph import Graph
from src.utils.config_parser import Config
from src.utils.create_graph import create_homophilic_graph2, create_heterophilic_graph2

dev = os.environ.get("device", "cpu")
device = torch.device(dev)

path = os.environ.get("CONFIG_PATH")
config = Config(path)


def define_graph(dataset_name=config.dataset.dataset_name):
    root = f"./datasets/{dataset_name}"
    os.makedirs(root, exist_ok=True)
    try:
        dataset = None
        if dataset_name in ["Cora", "PubMed", "CiteSeer"]:
            dataset = Planetoid(root=root, name=dataset_name)
            node_ids = torch.arange(dataset[0].num_nodes)
            edge_index = dataset[0].edge_index
        elif dataset_name in ["chameleon", "crocodile", "squirrel"]:
            dataset = WikipediaNetwork(
                root=root, geom_gcn_preprocess=True, name=dataset_name
            )
            node_ids = torch.arange(dataset[0].num_nodes)
            edge_index = dataset[0].edge_index
        elif dataset_name in [
            "Roman-empire",
            "Amazon-ratings",
            "Minesweeper",
            "Tolokers",
            "Questions",
        ]:
            dataset = HeterophilousGraphDataset(root=root, name=dataset_name)
        elif dataset_name in ["Actor"]:
            dataset = Actor(root=root)
        elif config.dataset.dataset_name in ["Computers", "Photo"]:
            dataset = Amazon(root=root, name=dataset_name)
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

        edge_index = to_undirected(edge_index)
        edge_index = remove_self_loops(edge_index)[0]
        graph = Graph(
            x=dataset[0].x.to(device),
            y=dataset[0].y.to(device),
            edge_index=edge_index.to(device),
            node_ids=node_ids.to(device),
            keep_sfvs=True,
            dataset_name=dataset_name,
            train_mask=dataset[0].train_mask,
            val_mask=dataset[0].val_mask,
            test_mask=dataset[0].test_mask,
            num_classes=dataset.num_classes,
        )

    return graph
