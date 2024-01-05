import os
from copy import deepcopy
from operator import itemgetter

import torch
import numpy as np
from sklearn import model_selection
import torch.nn.functional as F
from torch_geometric.typing import OptTensor
from torch_geometric.utils import degree, add_self_loops
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor

from src.utils.config_parser import Config
from src.models.Node2Vec import find_node2vect_embedings
from src.utils.GDV import GDV
from utils.utils import find_neighbors_, obtain_a, estimate_a

dev = os.environ.get("device", "cpu")
device = torch.device(dev)


path = os.environ.get("CONFIG_PATH")
config = Config(path).structure_model


class Data:
    def __init__(
        self,
        x: OptTensor = None,
        y: OptTensor = None,
        dataset_name=None,
        **kwargs,
    ) -> None:
        # super().__init__(
        #     x=x,
        #     y=y,
        #     **kwargs,
        # )
        self.x = x
        self.y = y
        self.num_nodes = x.shape[0]
        self.num_features = x.shape[1]
        self.dataset_name = dataset_name

        self.train_mask = kwargs.get("train_mask", None)
        self.test_mask = kwargs.get("val_mask", None)
        self.val_mask = kwargs.get("test_mask", None)

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


class Graph(Data):
    def __init__(
        self,
        edge_index: OptTensor,
        x: OptTensor = None,
        edge_attr: OptTensor = None,
        y: OptTensor = None,
        pos: OptTensor = None,
        node_ids=None,
        keep_sfvs=False,
        dataset_name=None,
        **kwargs,
    ) -> None:
        super().__init__(
            x=x,
            y=y,
            # edge_index=new_edges,
            # edge_attr=edge_attr,
            # pos=pos,
            dataset_name=dataset_name,
            **kwargs,
        )
        if node_ids is None:
            node_ids = np.arange(len(x))

        self.node_ids = node_ids
        self.original_edge_index = edge_index
        node_map, new_edges = Graph.reindex_nodes(node_ids, edge_index)
        self.edge_index = new_edges
        self.node_map = node_map
        self.edge_attr = edge_attr
        self.pos = pos
        self.inv_map = {v: k for k, v in node_map.items()}
        self.num_edges = edge_index.shape[1]

        self.sfv_initialized = "None"
        self.keep_sfvs = keep_sfvs
        if self.keep_sfvs:
            self.sfvs = {}

        self.abar = None

    def get_edges(self):
        return self.original_edge_index

    def reindex_nodes(nodes, edges):
        node_map = {node.item(): ind for ind, node in enumerate(nodes)}
        new_edges = edges.to("cpu").numpy()

        new_edges = np.vstack(
            (
                itemgetter(*new_edges[0])(node_map),
                itemgetter(*new_edges[1])(node_map),
            )
        )

        new_edges = torch.tensor(new_edges, device=edges.device)

        return node_map, new_edges

    def obtain_a(self, num_layers, estimate=False):
        if estimate:
            self.abar = estimate_a(self.edge_index, self.num_nodes, num_layers)
        else:
            self.abar = obtain_a(self.edge_index, self.num_nodes, num_layers)

    def add_structural_features(
        self,
        structure_type="degree",
        num_structural_features=100,
    ):
        if (
            self.sfv_initialized
            == structure_type
            # and num_structural_features == self.num_structural_features
        ):
            return

        structural_features = None
        if self.keep_sfvs:
            if structure_type in self.sfvs.keys():
                structural_features = self.sfvs[structure_type]

        if structural_features is None:
            (
                node_degree,
                # node_neighbors,
                structural_features,
                # node_negative_samples,
            ) = Graph.add_structural_features_(
                self.get_edges(),
                self.num_nodes,
                self.node_ids,
                structure_type=structure_type,
                num_structural_features=num_structural_features,
                dataset_name=self.dataset_name,
                save=True,
            )
            if structure_type in ["degree", "GDV", "node2vec"]:
                if self.keep_sfvs:
                    self.sfvs[structure_type] = deepcopy(structural_features)
        else:
            (
                node_degree,
                # node_neighbors,
                _,
                # node_negative_samples,
            ) = Graph.add_structural_features_(
                self.get_edges(),
                self.num_nodes,
                self.node_ids,
                structure_type="None",
                num_structural_features=num_structural_features,
            )

        if structure_type in ["degree", "GDV", "node2vec"]:
            self.sfv_initialized = deepcopy(structure_type)

        self.structural_features = structural_features
        self.degree = node_degree
        # self.node_neighbors = node_neighbors
        # self.negative_samples = node_negative_samples
        self.num_structural_features = num_structural_features

    def add_structural_features_(
        edge_index,
        num_nodes=None,
        node_ids=None,
        structure_type="degree",
        num_structural_features=100,
        dataset_name=None,
        save=False,
    ):
        if num_nodes is None:
            num_nodes = max(torch.flatten(edge_index)) + 1
        if node_ids is None:
            node_ids = np.arange(num_nodes)
        # node_neighbors = []
        # # node_negative_samples = []
        # for node_id in node_ids:
        #     neighbors = find_neighbors_(node_id, edge_index)

        #     node_neighbors.append(neighbors)

        node_degree = degree(edge_index[0]).long()

        structural_features = None
        if structure_type == "degree":
            structural_features = Graph.calc_degree_features(
                edge_index, num_nodes, num_structural_features
            )
        elif structure_type == "GDV":
            if dataset_name is not None:
                path = (
                    f"models/{dataset_name}/{structure_type}/{structure_type}_model.pkl"
                )
                if os.path.exists(path):
                    structural_features = torch.load(path)
            if structural_features is None:
                structural_features = Graph.calc_GDV(edge_index)
                if save:
                    if dataset_name is not None:
                        directory = f"models/{dataset_name}/{structure_type}/"
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        path = f"{directory}{structure_type}_model.pkl"
                        torch.save(structural_features, path)
        elif structure_type == "node2vec":
            if dataset_name is not None:
                path = (
                    f"models/{dataset_name}/{structure_type}/{structure_type}_model.pkl"
                )
                if os.path.exists(path):
                    structural_features = torch.load(path)
            if structural_features is None:
                structural_features = find_node2vect_embedings(
                    edge_index,
                    embedding_dim=num_structural_features,
                )
                if save:
                    if dataset_name is not None:
                        directory = f"models/{dataset_name}/{structure_type}/"
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        path = f"{directory}{structure_type}_model.pkl"
                        torch.save(structural_features, path)
        elif structure_type == "mp":
            d = Graph.calc_degree_features(
                edge_index, num_nodes, num_structural_features
            )
            structural_features = Graph.calc_mp(
                edge_index,
                d,
                iteration=config.num_mp_vectors,
            )
        elif structure_type == "random":
            structural_features = Graph.initialize_random_features(
                size=(len(node_ids), num_structural_features)
            )
        else:
            structural_features = None

        return node_degree, structural_features

    def calc_degree_features(edge_index, num_nodes, num_structural_features=100):
        node_degree1 = degree(edge_index[0], num_nodes).float()
        node_degree2 = degree(edge_index[1], num_nodes).float()
        node_degree = torch.round((node_degree1 + node_degree2) / 2).long()
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

    def initialize_random_features(size):
        return torch.normal(
            0,
            0.05,
            size=size,
            requires_grad=True,
        )

    def reset_parameters(self) -> None:
        if config.structure_type == "random":
            self.structural_features = Graph.initialize_random_features(
                size=self.structural_features.shape
            )

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
