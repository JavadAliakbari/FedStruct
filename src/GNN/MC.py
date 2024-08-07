import os
import sys

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

from tqdm import tqdm

from src import *

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

import torch
from torch_geometric.utils import (
    to_dense_adj,
)

# from src import *
from src.utils.graph import Graph
from src.utils.define_graph import define_graph
from src.utils.graph_partitioning import partition_graph


def cut_S(S, eps):
    for i in range(S.shape[0]):
        if S[i] > eps:
            S[i] -= eps
        elif S[i] < -eps:
            S[i] += eps
        else:
            S[i] = 0

    return S


def estimate(graph: Graph, n):
    # n = graph.num_nodes
    ps = 2 * graph.num_nodes / n - (graph.num_nodes**2) / (n**2)
    p = 1.2172 + 1.8588 * ps
    e1 = 4e-6
    e2 = 1e-6
    D = to_dense_adj(graph.edge_index, max_num_nodes=n)[0]
    u = 1 / torch.norm(D)
    Y = torch.zeros((n, n), dtype=torch.float32)
    E = torch.zeros((n, n), dtype=torch.float32)

    bar = tqdm(total=100)

    for i in range(100):
        U, S, V = torch.svd(D - E + Y / u)
        S2 = cut_S(S, 1 / u)
        A = torch.matmul(torch.matmul(U, torch.diag(S2)), V.T)
        E2 = D - A + Y / u
        E2[graph.node_ids, :] = 0
        E2[:, graph.node_ids] = 0
        Y += u * (D - A - E2)
        dE = E2 - E
        dS = min(u, torch.sqrt(u)) * torch.norm(dE) / torch.norm(D)
        dF = torch.norm(D - A - E) / torch.norm(D)
        E = E2
        # D2 = A + E - Y / u

        # ss = torch.sum(torch.abs(D1 - D2))
        bar.set_postfix(
            {
                "ds": dS.item(),
                "df": dF.item(),
                # "da": ss.item(),
            }
        )
        bar.update()

        if dF < e1 and dS < e2:
            return A, E

        if dS < e2:
            u = p * u

        # u = update_u(u, p, dE, D, e2)

    return A, E


if __name__ == "__main__":
    graph1 = define_graph("Cora")
    subgraphs = partition_graph(
        graph1, config.subgraph.num_subgraphs, config.subgraph.partitioning
    )
    graph2 = subgraphs[0]

    n = graph1.num_nodes
    D1 = to_dense_adj(graph1.edge_index, max_num_nodes=n)[0]
    A, E = estimate(graph2, graph1.num_nodes)

    D2 = A + E

    print(torch.sum(torch.abs(D1 - D2)))
