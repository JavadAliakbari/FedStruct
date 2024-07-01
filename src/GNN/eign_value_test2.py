import os
import sys

import torch

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

from torch_geometric.utils import to_dense_adj, get_laplacian
import matplotlib.pyplot as plt

from src.utils.define_graph import define_graph


def estimate_eigh(A, m):
    T, V = Lanczos_func(A, m)
    D_, U_ = torch.linalg.eigh(T)
    e = torch.diff(D_)
    th = torch.min(0.1 * torch.abs(D_[:-1]), torch.full((D_.shape[0] - 1,), 0.01))
    mask = torch.cat((torch.abs(e) - th > 0, torch.tensor([True])))
    D_2 = D_[mask]
    U_2 = U_[:, mask]
    U2 = torch.matmul(V, U_2)

    # U2 = torch.matmul(V, U_)
    # D_2 = D_

    return D_2, U2


def Lanczos_func(A, m=10):
    n = A.shape[0]
    B = torch.zeros(m - 1)
    a = torch.zeros(m)
    V = torch.zeros((n, m))
    v = torch.rand(n)
    v = v / torch.norm(v)
    V[:, 0] = v
    wp = torch.einsum("ij,j->i", A, V[:, 0])
    a[0] = torch.einsum("i,i", wp, V[:, 0])
    w = wp - a[0] * V[:, 0]
    for j in range(1, m):
        B[j - 1] = torch.norm(w)
        if B[j - 1] != 0:
            V[:, j] = w / B[j - 1]
        else:
            print("wooooo\n")
            v = torch.rand(n)
            v = v / torch.norm(v)
            V[:, j] = v
        wp = torch.einsum("ij,j->i", A, V[:, j])
        a[j] = torch.einsum("i,i", wp, V[:, j])
        w = wp - a[j] * V[:, j] - B[j - 1] * V[:, j - 1]

    T = torch.diag(a) + torch.diag(B, 1) + torch.diag(B, -1)

    return T, V


# Example of creating sparse and low-rank matrices
# n = 100
# rank = 23
# m = 50

# # Create low-rank matrices
# U_A = np.random.rand(n, rank)
# V_A = np.random.rand(rank, n)
# A = U_A @ V_A
# A = np.matmul(A, A.T) / n / n
if __name__ == "__main__":
    dataset_name = "Cora"
    graph = define_graph(dataset_name)

    n = graph.num_nodes
    m = n
    # m = 1000
    Lap_edges, lap_weights = get_laplacian(graph.edge_index)
    A = to_dense_adj(Lap_edges, edge_attr=lap_weights)[0]
    # A = A.numpy()

    D, U = torch.linalg.eigh(A)
    A2 = torch.matmul(U, torch.matmul(torch.diag(D), U.T))

    D_2, U2 = estimate_eigh(A, m)

    # T2 = np.matmul(U_2, np.matmul(np.diag(D_2), U_2.T))

    diff = torch.abs(D.unsqueeze(-1) - D_2)
    idx = torch.argmin(diff, axis=0)
    D2 = D[idx]

    A3 = torch.matmul(U2, torch.matmul(torch.diag(D_2), U2.T))
    print(torch.sum(torch.abs(D_2 - D2)))

    # plt.plot(D, marker="*")
    plt.plot(D2, marker="*")
    plt.plot(D_2, marker=".")
    plt.show()

    a = 2
