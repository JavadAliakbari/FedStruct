import os
from copy import deepcopy
from operator import itemgetter

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor
from torch_geometric.utils import (
    degree,
    to_undirected,
    scatter,
    remove_self_loops,
)

from src import *
from src.GNN.Lanczos import estimate_eigh
from src.models.GDV import GDV
from src.utils.data import Data
from src.models.Node2Vec import find_node2vect_embedings
from src.utils.utils import create_rw, find_neighbors_, obtain_a

dataset_name = config.dataset.dataset_name


class AGraph(Data):
    def __init__(
        self,
        abar: torch.Tensor | SparseTensor,
        x: torch.Tensor | SparseTensor | None = None,
        y: torch.Tensor | None = None,
        node_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(x, y, node_ids, **kwargs)
        self.abar = abar


class Graph(Data):
    def __init__(
        self,
        edge_index: torch.Tensor | None = None,
        x: torch.Tensor | SparseTensor | None = None,
        edge_attr: torch.Tensor | SparseTensor | None = None,
        y: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        node_ids=None,
        keep_sfvs=False,
        **kwargs,
    ) -> None:
        if node_ids is None:
            node_ids = torch.arange(len(x))
        super().__init__(
            x=x,
            y=y,
            node_ids=node_ids,
            **kwargs,
        )

        self.original_edge_index = edge_index
        node_map, new_edges = Graph.reindex_nodes(node_ids, edge_index)
        self.edge_index = new_edges
        self.node_map = node_map
        self.edge_attr = edge_attr
        self.pos = pos
        self.inv_map = {v: k for k, v in node_map.items()}
        self.num_edges = edge_index.shape[1]

        self.inter_edges = kwargs.get("inter_edges", None)
        self.external_nodes = kwargs.get("external_nodes", None)

        self.keep_sfvs = keep_sfvs
        if self.keep_sfvs:
            self.sfvs = {}

        self.abar = None
        self.structural_features = None

    def get_edges(self):
        # return only the original intra edges
        return self.original_edge_index

    def get_all_edges(self):
        # return both intra and inter connections
        if self.inter_edges is not None:
            return torch.concat((self.original_edge_index, self.inter_edges), dim=1)
        return self.get_edges()

    def reindex_nodes(nodes, edges):
        node_map = {node.item(): ind for ind, node in enumerate(nodes)}
        if edges.shape[1] == 0:
            new_edges = torch.empty((2, 0), dtype=torch.int64, device=edges.device)
        else:
            new_edges = edges.cpu().numpy()

            new_edges = np.vstack(
                (
                    itemgetter(*new_edges[0])(node_map),
                    itemgetter(*new_edges[1])(node_map),
                )
            )

            new_edges = torch.tensor(new_edges, dtype=torch.int64, device=edges.device)

        return node_map, new_edges

    def obtain_a(
        self,
        num_layers=config.structure_model.DGCN_layers,
        estimate=False,
        pruning=False,
    ):
        if self.abar is None:
            self.abar = obtain_a(
                self.edge_index, self.num_nodes, num_layers, estimate, pruning
            )

    def add_structural_features(
        self,
        structure_type="degree",
        num_structural_features=100,
        num_spectral_features=None,
    ):
        structural_features = None
        if self.keep_sfvs:
            if structure_type in self.sfvs.keys():
                structural_features = self.sfvs[structure_type]

        if structural_features is None:
            structural_features = Graph.add_structural_features_(
                self.get_edges(),
                self.num_nodes,
                structure_type=structure_type,
                num_structural_features=num_structural_features,
                num_spectral_features=num_spectral_features,
                save=True,
            )
            if structure_type in ["degree", "GDV", "node2vec"]:
                if self.keep_sfvs:
                    self.sfvs[structure_type] = deepcopy(structural_features)

        self.structural_features = structural_features
        self.num_structural_features = num_structural_features

    def add_structural_features_(
        edge_index,
        num_nodes=None,
        structure_type="degree",
        num_structural_features=100,
        num_spectral_features=None,
        save=False,
    ):
        if num_nodes is None:
            num_nodes = max(torch.flatten(edge_index)) + 1

        directory = f"models/{dataset_name}/{structure_type}/"
        path = f"{directory}{structure_type}_model.pkl"
        if os.path.exists(path):
            structural_features = torch.load(path)
            return structural_features

        if structure_type == "degree":
            structural_features = Graph.calc_degree_features(
                edge_index, num_nodes, num_structural_features
            )
        elif structure_type == "GDV":
            structural_features = Graph.calc_GDV(edge_index)
        elif structure_type == "node2vec":
            structural_features = find_node2vect_embedings(
                edge_index, embedding_dim=num_structural_features
            )
        elif structure_type == "mp":
            structural_features = Graph.calc_mp(
                edge_index,
                num_nodes,
                num_structural_features,
                iteration=config.structure_model.num_mp_vectors,
            )
        elif structure_type == "hop2vec":
            if num_spectral_features is None:
                num_spectral_features = num_nodes
            structural_features = Graph.initialize_random_features(
                size=(num_spectral_features, num_structural_features)
            )
        elif structure_type == "fedstar":
            structural_features = Graph.calc_fedStar(
                edge_index, num_nodes, num_structural_features
            )
        else:
            structural_features = None

        if save and structure_type in ["GDV", "node2vec", "fedstar"]:
            os.makedirs(directory, exist_ok=True)
            torch.save(structural_features, path)

        return structural_features

    def calc_degree_features(edge_index, num_nodes, size=100):
        node_degree1 = degree(edge_index[0], num_nodes).float()
        node_degree2 = degree(edge_index[1], num_nodes).float()
        node_degree = torch.round((node_degree1 + node_degree2) / 2).long()
        clipped_degree = torch.clip(node_degree, 0, size - 1)
        structural_features = F.one_hot(clipped_degree, size).float()

        return structural_features

    def calc_GDV(edge_index):
        gdv = GDV()
        structural_features = gdv.count5(edges=edge_index)
        sc = StandardScaler()
        structural_features = sc.fit_transform(structural_features)
        structural_features = torch.tensor(structural_features, dtype=torch.float32)

        return structural_features

    def calc_mp(edge_index, num_nodes, size=100, iteration=10):
        degree = Graph.calc_degree_features(edge_index, num_nodes, size)
        message_passing = MessagePassing(aggr="sum")
        sc = StandardScaler()

        x = degree
        mp = [x]
        for _ in range(iteration - 1):
            x = message_passing.propagate(edge_index, x=x)
            y = sc.fit_transform(x.numpy())
            mp.append(torch.tensor(y))

        mp = torch.sum(torch.stack(mp), dim=0)
        return mp

    def calc_fedStar(edge_index, num_nodes, size=100):
        SE_rw = create_rw(edge_index, num_nodes, config.structure_model.rw_len)
        SE_dg = Graph.calc_degree_features(
            edge_index, num_nodes, size - config.structure_model.rw_len
        )
        SE_rw_dg = torch.cat([SE_rw, SE_dg], dim=1)

        return SE_rw_dg

    def initialize_random_features(size):
        return torch.normal(
            0,
            0.05,
            size=size,
            requires_grad=True,
        )

    def reset_parameters(self) -> None:
        if config.structure_model.structure_type == "hop2vec":
            self.structural_features = Graph.initialize_random_features(
                size=self.structural_features.shape
            )

    def find_neighbors(self, node_id, include_node=False, include_external=False):
        if include_external:
            edges = torch.concat((self.get_edges(), self.inter_edges), dim=1)
        else:
            edges = self.get_edges()

        return find_neighbors_(
            node_id=node_id,
            edge_index=edges,
            include_node=include_node,
        )

    def create_L(self, normalization=None):
        nodes = self.node_ids

        num_nodes = self.x.shape[0]
        intra_edges = self.original_edge_index
        inter_edges = self.inter_edges
        if inter_edges is not None:
            edges = torch.concat((intra_edges, inter_edges), dim=1)
        else:
            edges = intra_edges

        # prob_tensor = torch.full((edges.shape[1],), 0.50)
        # selected_edges = torch.bernoulli(prob_tensor).bool()
        # edges = edges[:, selected_edges]

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

    def calc_eignvalues(self, estimate=False):
        if estimate:
            self.D, self.U = estimate_eigh(self.L, config.spectral.lanczos_iter)
            # self.D, self.U = estimate_eigh(self.L, num_nodes)
        else:
            self.D, self.U = torch.linalg.eigh(self.L)

        if config.spectral.spectral_len > 0:
            self.U = self.U[:, : config.spectral.spectral_len]
            self.D = self.D[: config.spectral.spectral_len]
