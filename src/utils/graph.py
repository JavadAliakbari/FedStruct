from ast import Dict
from operator import itemgetter

import torch
import numpy as np
from sklearn import model_selection
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor, Tensor
from torch_geometric.utils import degree
from sklearn.preprocessing import StandardScaler, normalize
from torch_geometric.nn import MessagePassing

from src.utils.config_parser import Config
from src.models.Node2Vec import find_node2vect_embedings
from src.utils.GDV2 import GDV


config = Config()


class Graph(Data):
    def __init__(
        self,
        edge_index: OptTensor,
        x: OptTensor = None,
        edge_attr: OptTensor = None,
        y: OptTensor = None,
        pos: OptTensor = None,
        node_ids=None,
        **kwargs,
    ) -> None:
        if node_ids is None:
            node_ids = np.arange(len(x))

        node_map, new_edges = Graph.reindex_nodes(node_ids, edge_index)
        super().__init__(
            x=x,
            edge_index=new_edges,
            edge_attr=edge_attr,
            y=y,
            pos=pos,
            **kwargs,
        )
        self.node_ids = node_ids
        self.node_map = node_map
        self.inv_map = {v: k for k, v in node_map.items()}

    def get_edges(self):
        new_edges = np.vstack(
            (
                itemgetter(*np.array(self.edge_index[0]))(self.inv_map),
                itemgetter(*np.array(self.edge_index[1]))(self.inv_map),
            )
        )

        return torch.tensor(new_edges)

    def reindex_nodes(nodes, edges):
        node_map = {node.item(): ind for ind, node in enumerate(nodes)}
        # node_map = dict.fromkeys(nodes, np.arange(len(nodes)))
        # new_edges = np.vstack((node_map[edges[0, :]], node_map[edges[1, :]]))
        new_edges = np.vstack(
            (
                itemgetter(*np.array(edges[0]))(node_map),
                itemgetter(*np.array(edges[1]))(node_map),
            )
        )

        return node_map, torch.tensor(new_edges)

    def get_masks(self):
        return (self.train_mask, self.val_mask, self.test_mask)

    def set_masks(self, masks):
        self.train_mask = masks[0]
        self.val_mask = masks[1]
        self.test_mask = masks[2]

    def add_masks(self, train_size=0.5, test_size=0.2):
        num_nodes = self.num_nodes
        indices = torch.arange(num_nodes)

        train_indices, test_indices = model_selection.train_test_split(
            indices,
            train_size=train_size,
            test_size=test_size,
        )

        self.train_mask = torch.isin(indices, train_indices)
        self.test_mask = torch.isin(indices, test_indices)
        self.val_mask = ~(self.test_mask | self.train_mask)

    def add_structural_features(
        self,
        structure_type="degree",
        num_structural_features=100,
    ):
        (
            node_degree,
            node_neighbors,
            structural_features,
            node_negative_samples,
        ) = Graph.add_structural_features_(
            self.get_edges(),
            self.node_ids,
            structure_type=structure_type,
            num_structural_features=num_structural_features,
        )

        self.structural_features = structural_features
        self.degree = node_degree
        self.node_neighbors = node_neighbors
        self.negative_samples = node_negative_samples
        self.num_structural_features = num_structural_features

    def add_structural_features_(
        edge_index,
        node_ids=None,
        structure_type="degree",
        num_structural_features=100,
    ):
        if node_ids is None:
            node_ids = np.arange(max(torch.flatten(edge_index)) + 1)
        node_neighbors = []
        node_negative_samples = []
        for node_id in node_ids:
            neighbors = Graph.find_neighbors_(node_id, edge_index)
            negative_samples = Graph.find_negative_samples(node_ids, neighbors)

            node_neighbors.append(neighbors)
            node_negative_samples.append(negative_samples)

        node_degree = degree(edge_index[0]).long()

        if structure_type == "degree":
            structural_features = Graph.calc_degree_features(
                edge_index, num_structural_features
            )
        elif structure_type == "GDV":
            structural_features = Graph.calc_GDV(edge_index)
        elif structure_type == "node2vec":
            structural_features = find_node2vect_embedings(
                edge_index,
                embedding_dim=num_structural_features,
                epochs=50,
            )
        elif structure_type == "mp":
            d = Graph.calc_degree_features(edge_index, num_structural_features)
            structural_features = Graph.calc_mp(
                edge_index,
                d,
                iteration=config.structure_model.num_mp_vectors,
            )
        # elif structure_type == "struc2vec":
        #     structural_features = Graph.calc_stuc2vec()

        return node_degree, node_neighbors, structural_features, node_negative_samples

    def calc_degree_features(edge_index, num_structural_features=100):
        node_degree = degree(edge_index[0]).long()
        clipped_degree = torch.clip(node_degree, 0, num_structural_features - 1)
        structural_features = F.one_hot(clipped_degree, num_structural_features).float()

        return structural_features

    def calc_GDV(edge_index):
        gdv = GDV()
        structural_features = gdv.count5(edges=edge_index)
        sc = StandardScaler()
        structural_features = sc.fit_transform(structural_features)
        structural_features = torch.tensor(structural_features, dtype=torch.float32)

        return structural_features

    def calc_mp(edge_index, degree, iteration=10):
        message_passing = MessagePassing(aggr="sum")

        x = degree
        mp = []
        mp.append(x.numpy())
        for _ in range(iteration - 1):
            x = message_passing.propagate(edge_index, x=x)
            mp.append(x.numpy())

        mp = np.array(mp, dtype=np.float32).transpose([1, 2, 0])
        mp = torch.tensor(mp)
        return mp

    # def calc_stuc2vec():
    #     with open(f"chameleon.emb", "r") as f:
    #         lines = []
    #         for line in f:
    #             lines.append([float(x) for x in line.split()])
    #         # lines = f.readlines()
    #         x = np.zeros((int(lines[0][0]), int(lines[0][1])), dtype=np.float32)
    #         for line in lines[1:]:
    #             x[int(line[0])] = np.array(line[1:])

    #     return torch.Tensor(x)

    def find_class_neighbors(self):
        class_groups, negative_class_groups = Graph.find_class_neighbors_(
            self.get_edges(),
            self.y,
            self.train_mask,
            self.node_ids,
        )

        self.class_groups = class_groups
        self.negative_class_groups = negative_class_groups

    def find_class_neighbors_(edge_index, y, train_mask, node_ids=None):
        if node_ids is None:
            node_ids = np.arange(max(torch.flatten(edge_index)) + 1)

        classes = torch.unique(y).numpy()
        class_groups = {}
        negative_class_groups = {}
        for class_id in classes:
            class_groups[class_id] = node_ids[y == class_id & train_mask]
            negative_class_groups[class_id] = node_ids[y != class_id & train_mask]

        return class_groups, negative_class_groups

    def find_negative_samples(node_ids, neighbors):
        neighbor_nodes_mask = np.isin(node_ids, neighbors)
        other_nodes = node_ids[~neighbor_nodes_mask]

        return np.array(other_nodes)

    def find_neigbors(self, node_id, include_node=False):
        return Graph.find_neighbors_(
            node_id=node_id,
            edge_index=self.edge_index,
            node_map=self.node_map,
            include_node=include_node,
        )

    def find_neighbors_(
        node_id: int,
        edge_index: Tensor,
        node_map: Dict = None,
        include_node=False,
    ):
        if node_map is not None:
            new_node_id = node_map[node_id]
        else:
            new_node_id = node_id
        all_neighbors = np.unique(
            np.hstack(
                (
                    edge_index[1, edge_index[0] == new_node_id],
                    edge_index[0, edge_index[1] == new_node_id],
                )
            )
        )

        if not include_node:
            all_neighbors = np.setdiff1d(all_neighbors, new_node_id)

        if len(all_neighbors) == 0:
            return all_neighbors

        if node_map is not None:
            inv_map = {v: k for k, v in node_map.items()}
            res = itemgetter(*all_neighbors)(inv_map)
            if len(all_neighbors) == 1:
                return [res]
            else:
                return list(res)
        else:
            return all_neighbors
