import os.path as osp

import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.nn.models import Node2Vec
from torch_geometric.datasets import TUDataset, Planetoid

from src.utils import config
from src.utils.graph import Graph
from src.models.GNN_models import MLP
from src.models.Node2Vec import find_node2vect_embedings
from src.utils.create_graph import (
    create_heterophilic_graph,
    create_heterophilic_graph2,
    create_homophilic_graph,
    create_homophilic_graph2,
)


if __name__ == "__main__":
    num_nodes = 5000
    mean_degree = 6
    num_edges = int(num_nodes * mean_degree / 2)
    num_patterns = int(0.08 * num_nodes)
    # num_classes = 2

    dataset = Planetoid(root="/tmp/Cora", name="Cora")

    node_ids = torch.arange(dataset[0].num_nodes)
    graph = Graph(
        x=dataset[0].x,
        y=dataset[0].y,
        edge_index=dataset[0].edge_index,
        node_ids=node_ids,
    )

    # graph = create_homophilic_graph2(num_patterns)
    # graph = create_heterophilic_graph2(num_patterns)
    # graph = create_heterophilic_graph(num_nodes, num_edges, num_patterns)
    graph.add_masks()

    masks = (graph.train_mask, graph.val_mask, graph.test_mask)
    y = graph.y.long()
    num_classes = max(y) + 1

    z = find_node2vect_embedings(graph.edge_index, 50)

    cls = MLP([64, 32, num_classes], dropout=0.1)

    cls.fit(z, y, masks, 200, verbose=True)

    tsne_model = TSNE(n_components=2, random_state=4, perplexity=25, n_jobs=7)
    x_embed = tsne_model.fit_transform(z)

    cmap = plt.get_cmap("gist_rainbow", num_classes)
    colors = [cmap(1.0 * i / num_classes) for i in range(num_classes)]

    fig, ax = plt.subplots()
    for i in range(num_classes):
        mask = y == i
        class_points = x_embed[mask]
        ax.scatter(
            class_points[:, 0],
            class_points[:, 1],
            color=colors[i],
            marker=f"${i}$",
            label=i,
        )

    ax.legend(loc="lower right")

    plt.show()
