import os
import random

import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import (
    TUDataset,
    Planetoid,
    WikipediaNetwork,
    HeterophilousGraphDataset,
)
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops


from src.server import Server
from src.utils.create_graph import create_heterophilic_graph2, create_homophilic_graph2
from src.utils.graph import Graph
from src.utils.logger import get_logger
from src.utils.config_parser import Config
from src.models.GNN_models import MLP

config = Config()

random.seed(4)
np.random.seed(4)
torch.manual_seed(4)


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


if __name__ == "__main__":
    save_path = f"./results/{config.dataset.dataset_name}/{config.structure_model.structure_type}/structure/"
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

    graph.add_masks()

    y = graph.y.long()
    num_classes = max(y).item() + 1
    MLP_server = Server(
        graph, num_classes, classifier_type="MLP", save_path=save_path, logger=_LOGGER
    )
    GNN_server = Server(
        graph, num_classes, classifier_type="GNN", save_path=save_path, logger=_LOGGER
    )

    GNN_server.train_sd_predictor(100, plot=True, predict=True)
    GNN_server.test_sd_predictor()

    x = GNN_server.sd_predictor.graph.structural_features
    graph.x = x
    graph.num_features = x.shape[1]

    MLP_server.train_local_classifier(200)
    _LOGGER.info(f"Server MLP test accuracy: {MLP_server.test_local_classifier()}")
    GNN_server.train_local_classifier(500)
    _LOGGER.info(f"Server GNN test accuracy: {GNN_server.test_local_classifier()}")

    # x = server.get_sd_embeddings()

    message_passing = MessagePassing(aggr="mean")
    cls = MLP(
        layer_sizes=[
            config.structure_model.num_structural_features,
            32,
            num_classes,
        ],
        batch_normalization=True,
    )
    masks = (graph.train_mask, graph.val_mask, graph.test_mask)
    # train_x = x[graph.train_mask]
    # train_y = y[graph.train_mask]
    # val_x = x[graph.val_mask]
    # val_y = y[graph.val_mask]
    test_y = y[graph.test_mask]

    edge_index = graph.edge_index
    edge_index = add_self_loops(edge_index)[0]

    test_acc_list = []
    l = 100
    for i in range(l):
        cls.reset_parameters()
        cls.fit(x, y, masks, epochs=100, verbose=False)
        test_x = x[graph.test_mask]
        test_acc = cls.test(test_x, test_y)

        _LOGGER.info(f"epoch: {i} test accuracy: {test_acc}")
        test_acc_list.append(test_acc)
        x = message_passing.propagate(edge_index, x=x)
        if i in [0, 2, 5, 10, 20, l - 1]:
            plot(x, round=i)

    # x = GNN_server.sd_predictor.graph.structural_features
    plt.figure()
    plt.plot(
        range(l), test_acc_list, label=config.structure_model.structure_type, marker="*"
    )
    plt.title(f"{config.dataset.dataset_name} Message Passing accuracy per round")
    plt.xlabel("round")
    plt.ylabel("accuracy")
    plt.legend()
    # plt.savefig(f"./Message Passing accuracy per round {config.dataset.dataset_name}")
    plt.savefig(f"{save_path}Message Passing accuracy per round.png")

    plt.show()
