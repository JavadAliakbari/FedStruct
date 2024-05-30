import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import (
    HeterophilousGraphDataset,
    Planetoid,
    TUDataset,
    WikipediaNetwork,
)
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected
from sklearn.preprocessing import StandardScaler

# from src.utils.utils import *
from src.models.model_binders import MLP
from src.models.Node2Vec import find_node2vect_embedings
from src.utils.create_graph import create_heterophilic_graph2, create_homophilic_graph2
from src.utils.graph import Graph
from src.utils.logger import get_logger


def plot(x, round=0):
    tsne_model = TSNE(n_components=2, random_state=4, perplexity=20, n_jobs=7)
    x_embed = tsne_model.fit_transform(x)
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
    plt.title(f"TSNE distinction at round {round}")

    plt.savefig(f"{save_path}TSNE distinction at round {round}.png")
    plt.close()


def log_config():
    _LOGGER.info(f"dataset name: {config.dataset.dataset_name}")
    _LOGGER.info(f"learning rate: {config.model.lr}")
    _LOGGER.info(f"weight decay: {config.model.weight_decay}")
    _LOGGER.info(f"dropout: {config.model.dropout}")
    _LOGGER.info(f"mlp layer sizes: {config.feature_model.mlp_layer_sizes}")
    _LOGGER.info(
        f"num structural features: {config.structure_model.num_structural_features}"
    )


if __name__ == "__main__":
    save_path = f"./results/{config.dataset.dataset_name}/message passing/"
    _LOGGER = get_logger(
        name=f"SD_{config.dataset.dataset_name}_{config.structure_model.structure_type}",
        log_on_file=True,
        save_path=save_path,
    )
    try:
        dataset = None
        if config.dataset.dataset_name in ["Cora", "PubMed", "CiteSeer"]:
            dataset = Planetoid(
                root=f"/tmp/{config.dataset.dataset_name}",
                name=config.dataset.dataset_name,
            )
            node_ids = torch.arange(dataset[0].num_nodes)
            edge_index = dataset[0].edge_index
            num_classes = dataset.num_classes
        elif config.dataset.dataset_name in ["chameleon", "crocodile", "squirrel"]:
            dataset = WikipediaNetwork(
                root=f"/tmp/{config.dataset.dataset_name}",
                geom_gcn_preprocess=True,
                name=config.dataset.dataset_name,
            )
            node_ids = torch.arange(dataset[0].num_nodes)
            edge_index = dataset[0].edge_index
            num_classes = dataset.num_classes
        elif config.dataset.dataset_name in [
            "Roman-empire",
            "Amazon-ratings",
            "Minesweeper",
            "Tolokers",
            "Questions",
        ]:
            dataset = HeterophilousGraphDataset(
                root=f"/tmp/{config.dataset.dataset_name}",
                name=config.dataset.dataset_name,
            )
        elif config.dataset.dataset_name == "Heterophilic_example":
            num_patterns = 100
            graph = create_heterophilic_graph2(num_patterns, use_random_features=False)
        elif config.dataset.dataset_name == "Homophilic_example":
            num_patterns = 100
            graph = create_homophilic_graph2(num_patterns, use_random_features=False)

    except:
        _LOGGER.info("dataset name does not exist!")

    if dataset is not None:
        node_ids = torch.arange(dataset[0].num_nodes)
        edge_index = dataset[0].edge_index
        num_classes = dataset.num_classes

        edge_index = to_undirected(edge_index)
        edge_index = remove_self_loops(edge_index)[0]
        graph = Graph(
            x=dataset[0].x,
            y=dataset[0].y,
            edge_index=edge_index,
            node_ids=node_ids,
        )
    else:
        num_classes = max(graph.y).item() + 1

    log_config()

    graph.add_masks()

    y = graph.y.long()
    num_classes = max(y).item() + 1

    # x = GNN_server.sd_predictor.graph.structural_features

    message_passing = MessagePassing(aggr="mean")
    cls = MLP(
        layer_sizes=[config.structure_model.num_structural_features]
        + config.feature_model.mlp_layer_sizes
        + [num_classes],
        normalization="batch",
    )

    sc = StandardScaler()

    y_train = y[graph.train_mask]
    y_val = y[graph.val_mask]
    y_test = y[graph.test_mask]

    test_acc_list = {}
    plt.figure()
    for sd in ["GDV"]:
        # for sd in ["degree", "GDV", "node2vec"]:
        if sd == "degree":
            x = Graph.calc_degree_features(
                graph.edge_index, config.structure_model.num_structural_features
            )
        elif sd == "GDV":
            x = Graph.calc_GDV(graph.edge_index)
        elif sd == "node2vec":
            x = find_node2vect_embedings(
                edge_index,
                embedding_dim=config.structure_model.num_structural_features,
                epochs=25,
            )
        edge_index = graph.edge_index
        edge_index = add_self_loops(edge_index)[0]

        l = 100
        test_acc_list[sd] = []
        for i in range(l):
            x_train = x[graph.train_mask]
            x_val = x[graph.val_mask]
            x_test = x[graph.test_mask]

            cls.reset_parameters()
            cls_val_acc, cls_val_loss = cls.fit(
                x_train,
                y_train,
                x_val,
                y_val,
                epochs=150,
                verbose=True,
                plot=True,
            )
            test_acc = cls.test(x_test, y_test)

            _LOGGER.info(f"epoch: {i} test accuracy: {test_acc}")
            test_acc_list[sd].append(test_acc)
            x = message_passing.propagate(edge_index, x=x)
            plt.show()
            # x = sc.fit_transform(x)
            # x = torch.tensor(x, dtype=torch.float32)
            # if i in [0, 2, 5, 10, 20, l - 1]:
            #     plot(x, round=i)

        plt.plot(
            range(l),
            test_acc_list[sd],
            marker="*",
            label=sd,
        )

    plt.title(f"{config.dataset.dataset_name} Message Passing accuracy per round")
    plt.xlabel("round")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(f"{save_path}Message Passing accuracy per round.png")

    # plt.show()
