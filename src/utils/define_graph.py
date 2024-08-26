import os

import torch
from torch_geometric.transforms import FaceToEdge
from torch_geometric.datasets import (
    Planetoid,
    HeterophilousGraphDataset,
    WikipediaNetwork,
    Amazon,
    Actor,
    FAUST,
    JODIEDataset,
    EllipticBitcoinTemporalDataset,
    PPI,
)

from src import *
from src.utils.graph import Graph
from src.utils.create_graph import create_homophilic_graph2, create_heterophilic_graph2


def define_graph(dataset_name=config.dataset.dataset_name, **kwargs):
    root = f"./datasets/{dataset_name}"
    os.makedirs(root, exist_ok=True)
    # try:
    if True:
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
        elif dataset_name in ["PPI"]:
            split = kwargs.get("split", "train")
            dataset = PPI(root=root, split=split)
        elif config.dataset.dataset_name in ["Computers", "Photo"]:
            dataset = Amazon(root=root, name=dataset_name)
        elif dataset_name == "Heterophilic_example":
            num_patterns = 500
            graph = create_heterophilic_graph2(num_patterns, use_random_features=True)
        elif dataset_name == "Homophilic_example":
            num_patterns = 100
            graph = create_homophilic_graph2(num_patterns, use_random_features=True)
        elif dataset_name == "Faust":
            dataset = FAUST(root=root, transform=FaceToEdge)
        elif dataset_name == "EllipticBitcoin":
            t = kwargs.get("t", 10)
            dataset = EllipticBitcoinTemporalDataset(root=root, t=t)
        elif dataset_name in [
            "Reddit",
            "Wikipedia",
            "MOOC",
            "LastFM",
        ]:
            dataset = JODIEDataset(root=root, name=dataset_name)

    # except:
    #     # LOGGER.info("dataset name does not exist!")
    #     return None, 0

    if dataset is not None:
        data = dataset._data
        node_ids = torch.arange(data.num_nodes)
        edge_index = data.edge_index
        x = data.x
        if x is None:
            x = data.msg

        # edge_index = to_undirected(edge_index)
        # edge_index = remove_self_loops(edge_index)[0]

        graph = Graph(
            x=x.to(device),
            y=data.y.to(device),
            edge_index=edge_index.to(device),
            node_ids=node_ids.to(device),
            keep_sfvs=True,
            dataset_name=dataset_name,
            train_mask=data.get("train_mask", None),
            val_mask=data.get("val_mask", None),
            test_mask=data.get("test_mask", None),
            time=data.get("t", None),
            num_classes=dataset.num_classes,
        )

    return graph
