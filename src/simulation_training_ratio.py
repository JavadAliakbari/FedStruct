import os
import random
import json

import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.utils.utils import *
from src.define_graph import define_graph
from src.GNN_server import GNNServer
from src.MLP_server import MLPServer
from src.fedsage_server import FedSAGEServer
from src.utils.logger import get_logger
from src.utils.config_parser import Config
from src.simulation import *
from src.main import log_config

# Change plot canvas size
plt.rcParams["figure.figsize"] = [24, 16]
plt.rcParams["figure.dpi"] = 100  # 200 e.g. is really fine, but slower
plt.rcParams.update({"figure.max_open_warning": 0})

seed = 119
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

path = os.environ.get("CONFIG_PATH")
config = Config(path)


if __name__ == "__main__":
    graph, num_classes = define_graph(config.dataset.dataset_name)
    graph.obtain_a(config.structure_model.DGCN_layers)

    MLP_server = MLPServer(
        graph,
        num_classes,
    )

    GNN_server = GNNServer(
        graph,
        num_classes,
    )

    FedSage_server = FedSAGEServer(
        graph,
        num_classes,
    )

    rep = 10

    for partitioning in [config.subgraph.partitioning]:
        # for partitioning in ["random", "louvian", "kmeans"]:
        # for num_subgraphs in [config.subgraph.num_subgraphs]:
        for num_subgraphs in [5, 10, 20]:
            for train_ratio in [config.subgraph.train_ratio]:
                # for train_ratio in np.arange(0.1, 0.65, 0.05):
                test_ratio = config.subgraph.test_ratio
                # test_ratio = (1 - train_ratio) / 2
                epochs = config.model.epoch_classifier
                # epochs = int(train_ratio * 100 + 30)

                save_path = (
                    "./results/Paper Results/"
                    f"{config.dataset.dataset_name}/"
                    f"{partitioning}/"
                    f"{num_subgraphs}/"
                    f"{train_ratio}/"
                )
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                _LOGGER = get_logger(
                    name=f"average_{config.dataset.dataset_name}_{train_ratio}",
                    log_on_file=True,
                    save_path=save_path,
                )
                _LOGGER2 = get_logger(
                    name=f"all_{config.dataset.dataset_name}_{train_ratio}",
                    terminal=False,
                    log_on_file=True,
                    save_path=save_path,
                )
                log_config(_LOGGER)

                bar = tqdm(total=rep)
                results = []
                for i in range(rep):
                    result = run(
                        graph,
                        MLP_server,
                        GNN_server,
                        FedSage_server,
                        bar=bar,
                        epochs=epochs,
                        train_ratio=train_ratio,
                        test_ratio=test_ratio,
                        num_subgraphs=num_subgraphs,
                        partitioning=partitioning,
                    )
                    _LOGGER2.info(f"Run id: {i}")
                    _LOGGER2.info(json.dumps(result, indent=4))

                    results.append(result)

                    average_result = calc_average_std_result(results)
                    file_name = f"{save_path}final_result_{train_ratio}.csv"
                    save_average_result(average_result, file_name)

                    bar.update()

                _LOGGER.info(json.dumps(average_result, indent=4))

                _LOGGER.handlers.clear()
                _LOGGER2.handlers.clear()
