import torch
import numpy as np

from src.utils.GDV import GDV
from src.utils.graph import Graph
from src.utils.create_graph import (
    create_heterophilic_graph,
    create_heterophilic_graph2,
    create_homophilic_graph,
    create_homophilic_graph2,
)

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

    edges = torch.tensor(np.array(edges).T, dtype=int)

    num_patterns = 50
    edges = create_homophilic_graph2(num_patterns).edge_index
    gdv = GDV()
    orbit = gdv.count5(edges)
    a = 2
