import torch
from torch_geometric.utils import (
    to_undirected,
    scatter,
    remove_self_loops,
)

from src import *
from src.GNN.sGNN import SClassifier
from src.utils.graph import Graph
from src.GNN.eign_value_test2 import estimate_eigh


class SLaplace(SClassifier):
    def __init__(
        self,
        graph: Graph,
        hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
    ):
        super().__init__(graph, hidden_layer_size)
        self.create_L()

    def create_L(self, normalization=None):
        nodes = self.graph.node_ids

        num_nodes = self.graph.x.shape[0]
        intra_edges = self.graph.original_edge_index
        inter_edges = self.graph.inter_edges
        if inter_edges is not None:
            edges = torch.concat((intra_edges, inter_edges), dim=1)
        else:
            edges = intra_edges

        undirected_edges = to_undirected(edges)
        edge_mask = undirected_edges[0].unsqueeze(1).eq(nodes).any(1)
        directed_edges = undirected_edges[:, edge_mask]
        directed_edges, _ = remove_self_loops(directed_edges)

        nodes_ = nodes.unsqueeze(0)
        self_loops = torch.concat((nodes_, nodes_), dim=0)
        l_edge_index = torch.concat((directed_edges, self_loops), dim=1)

        edge_weight = torch.ones(
            directed_edges.size(1), dtype=torch.float32, device=directed_edges.device
        )
        row, col = directed_edges[0], directed_edges[1]
        deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce="sum")
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float("inf"), 0)

        if normalization is None:
            # D - A
            l_edge_weight = torch.concat(
                (-torch.ones_like(directed_edges[0]), deg[self_loops[0]])
            )
        elif normalization == "rw":
            # I - D^-1 A
            l_edge_weight = torch.concat(
                (-deg_inv[directed_edges[0]], torch.ones_like(nodes))
            )

        self.L = torch.sparse_coo_tensor(
            l_edge_index,
            l_edge_weight,
            (num_nodes, num_nodes),
            dtype=torch.float32,
            device=intra_edges.device,
        )

    def regularizer(self):
        x = self.graph.x
        r1 = torch.matmul(self.L, x)
        r = torch.matmul(x.T, r1)
        s = torch.trace(r) / torch.trace(torch.matmul(x.T, x))
        # s = torch.trace(r) / x.shape[1]

        return s


class SpectralLaplace(SClassifier):
    def __init__(
        self,
        graph: Graph,
        hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
    ):
        super().__init__(graph, hidden_layer_size)
        self.create_L()

    def create_L(self, normalization=None):
        nodes = self.graph.node_ids

        num_nodes = self.graph.x.shape[0]
        intra_edges = self.graph.original_edge_index
        inter_edges = self.graph.inter_edges
        if inter_edges is not None:
            edges = torch.concat((intra_edges, inter_edges), dim=1)
        else:
            edges = intra_edges

        undirected_edges = to_undirected(edges)
        edge_mask = undirected_edges[0].unsqueeze(1).eq(nodes).any(1)
        directed_edges = undirected_edges[:, edge_mask]
        directed_edges, _ = remove_self_loops(directed_edges)

        nodes_ = nodes.unsqueeze(0)
        self_loops = torch.concat((nodes_, nodes_), dim=0)
        l_edge_index = torch.concat((directed_edges, self_loops), dim=1)

        edge_weight = torch.ones(
            directed_edges.size(1), dtype=torch.float32, device=directed_edges.device
        )
        row, col = directed_edges[0], directed_edges[1]
        deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce="sum")
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float("inf"), 0)

        if normalization is None:
            # D - A
            l_edge_weight = torch.concat(
                (-torch.ones_like(directed_edges[0]), deg[self_loops[0]])
            )
        elif normalization == "rw":
            # I - D^-1 A
            l_edge_weight = torch.concat(
                (-deg_inv[directed_edges[0]], torch.ones_like(nodes))
            )

        self.L = torch.sparse_coo_tensor(
            l_edge_index,
            l_edge_weight,
            (num_nodes, num_nodes),
            dtype=torch.float32,
            device=intra_edges.device,
        ).to_dense()

        self.D, self.U = torch.linalg.eigh(self.L)

        # n = self.graph.x.shape[0]
        # self.D, self.U = estimate_eigh(self.L, 2000)
        # m = self.D.shape[0]
        # self.D = torch.concat((self.D, torch.zeros(n - m, dtype=torch.float32)))
        # self.U = torch.concat(
        #     (self.U, torch.zeros((self.U.shape[0], n - m), dtype=torch.float32)), dim=1
        # )

        # mask = abs(self.D) < 100
        # self.D[mask] = 0
        # self.U[:, mask] = 0

    def get_SFV(self):
        return torch.matmul(self.U, self.graph.x)

    def get_UD(self):
        return self.U, self.D

    def set_UD(self, U, D):
        self.U = U
        self.D = D

    def regularizer(self):
        x = self.graph.x
        r1 = torch.einsum("i,ij->ij", self.D, x)
        r = torch.matmul(x.T, r1)
        s = torch.trace(r) / torch.trace(torch.matmul(x.T, x))
        # s = torch.trace(r) / x.shape[1]

        return s

    def get_embeddings(self):
        # x = self.graph.x
        SFV = torch.matmul(self.U, self.graph.x)
        H = self.model(SFV)
        nodes = self.graph.node_ids
        # H = 0 * H1 + self.graph.x
        H = H[nodes]
        return H
