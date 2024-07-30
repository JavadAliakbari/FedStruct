from copy import deepcopy

import torch
from tqdm import tqdm

from src import *


def estimate_eigh(A, m, X=None, prune=False, block=False, p=5):
    if block:
        T, V = block_lanczos(A, X, m=m, p=p)
    else:
        T, V = Lanczos_func(A, m)
    D_, U_ = torch.linalg.eigh(T)
    if prune:
        e = torch.diff(D_)
        th = torch.min(0.1 * torch.abs(D_[:-1]), torch.full((D_.shape[0] - 1,), 0.01))
        mask = torch.cat((torch.abs(e) - th > 0, torch.tensor([True])))
        D_2 = D_[mask]
        U_2 = U_[:, mask]
    else:
        D_2 = D_
        U_2 = U_

    U2 = torch.matmul(V, U_2)
    U2 = U2.float()
    D_2 = D_2.float()

    # U2 = torch.matmul(V, U_)
    # D_2 = D_

    return D_2, U2


def Lanczos_func(A, m=10):
    n = A.shape[0]
    A = deepcopy(A).double()
    A = A.to_sparse()
    B = torch.zeros(m - 1, dtype=torch.double, device=dev)
    a = torch.zeros(m, dtype=torch.double, device=dev)
    V = torch.zeros((n, m), dtype=torch.double, device=dev)
    v = torch.randn(n, dtype=torch.double, device=dev)
    v = v / torch.norm(v)
    V[:, 0] = v
    wp = torch.matmul(A, V[:, 0]).to_dense()
    # wp = torch.einsum("ij,j->i", A, V[:, 0])
    a[0] = torch.einsum("i,i", wp, V[:, 0])
    w = wp - a[0] * V[:, 0]
    bar = tqdm(total=m - 1)
    for j in range(1, m):
        B[j - 1] = torch.norm(w)
        if B[j - 1] != 0:
            V[:, j] = w / B[j - 1]
        else:
            print("wooooo\n")
            v = torch.rand(n)
            v = v / torch.norm(v)
            V[:, j] = v
        wp = torch.matmul(A, V[:, j]).to_dense()
        # wp = torch.einsum("ij,j->i", A, V[:, j])
        a[j] = torch.einsum("i,i", wp, V[:, j])
        # a[j] = torch.einsum("i,i", wp - B[j - 1] * V[:, j - 1], V[:, j])
        w = wp - a[j] * V[:, j] - B[j - 1] * V[:, j - 1]

        # w -= V @ V.T @ w

        # T = torch.diag(a) + torch.diag(B, 1) + torch.diag(B, -1)

        # At = torch.matmul(V, torch.matmul(T, V.T))
        # e = torch.mean(torch.abs(A - At)).item()
        # bar.set_postfix({"e": e})
        bar.update()

    T = torch.diag(a) + torch.diag(B, 1) + torch.diag(B, -1)

    return T, V


def block_lanczos(H, X=None, m=10, p=5):
    n = H.shape[0]
    H = deepcopy(H).double()
    X = deepcopy(X).double()
    B = []
    A = []
    if X is None:
        X = torch.randn((n, p), dtype=torch.double, device=dev)
    else:
        p = X.shape[1]
    Qj, _ = torch.linalg.qr(X)
    Q = [Qj]

    bar = tqdm(total=m)
    for j in range(m):
        if j == 0:
            U = H @ Q[-1]
        else:
            U = H @ Q[-1] - Q[-2] @ B[-1].T

        Aj = Q[-1].T @ U
        A.append(Aj)
        R = U - Q[-1] @ Aj
        Qj, Bj = torch.qr(R)
        Q.append(Qj)
        B.append(Bj)

        bar.update()

    T = (
        torch.block_diag(*A)
        + torch.vstack(
            (
                torch.hstack(
                    (
                        torch.zeros((j * p, p), dtype=torch.double),
                        torch.block_diag(*B[:-1]),
                    )
                ),
                torch.zeros((p, (j + 1) * p), dtype=torch.double),
            )
        )
        + torch.vstack(
            (
                torch.zeros((p, (j + 1) * p), dtype=torch.double),
                torch.hstack(
                    (
                        torch.block_diag(*B[:-1]),
                        torch.zeros((j * p, p), dtype=torch.double),
                    )
                ),
            )
        )
    )
    V = torch.hstack(Q[:-1])

    return T, V
