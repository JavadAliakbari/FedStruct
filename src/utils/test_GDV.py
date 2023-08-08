import time

import torch
import numpy as np
from torch_geometric.datasets import TUDataset, Planetoid, WikipediaNetwork
from torch_geometric.utils import degree, to_undirected, remove_self_loops

from src.utils.config_parser import Config
from src.utils.GDV import GDV as GDV1

from src.utils.GDV2 import GDV as GDV2
from src.utils.graph import Graph
from src.utils.create_graph import (
    create_heterophilic_graph,
    create_heterophilic_graph2,
    create_homophilic_graph,
    create_homophilic_graph2,
)

config = Config()

if __name__ == "__main__":
    edges = [
        [0, 1],
        [1, 0],
        [0, 2],
        [2, 0],
        [0, 3],
        [3, 0],
        [0, 4],
        [4, 0],
        [0, 6],
        [6, 0],
        [1, 2],
        [2, 1],
        [3, 4],
        [4, 3],
        [4, 5],
        [5, 4],
        [5, 6],
        [6, 5],
    ]

    edges = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 3],
        [2, 4],
        [3, 4],
        [3, 5],
        [3, 6],
        [3, 7],
        [5, 6],
        [6, 7],
        [5, 8],
        [6, 8],
        [7, 8],
    ]

    edge_index = torch.tensor(np.array(edges).T, dtype=int)

    # num_patterns = 50
    # edge_index = create_homophilic_graph2(num_patterns).edge_index

    edge_index = Planetoid(
        root=f"/tmp/{config.dataset.dataset_name}", name=config.dataset.dataset_name
    ).edge_index
    # dataset = WikipediaNetwork(
    #     root=f"/tmp/{config.dataset.dataset_name}", geom_gcn_preprocess=True, name=config.dataset.dataset_name
    # )

    edge_index = to_undirected(edge_index)
    edge_index = remove_self_loops(edge_index)[0]

    gdv = GDV1()
    start = time.time()
    orbit1 = gdv.count5(edge_index)
    stop = time.time()
    time1 = stop - start
    print(f"time1: {time1}")

    gdv = GDV2()
    start = time.time()
    orbit2 = gdv.count5(edge_index)
    stop = time.time()
    time2 = stop - start
    print(f"time2: {time2}")

    print(sum(sum(abs(orbit1 - orbit2))))
    a = 2
