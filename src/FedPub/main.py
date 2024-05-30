from datetime import datetime

from src import *
from src.FedPub.utils import *
from src.FedPub.fedpub_server import FedPubServer
from src.utils.define_graph import define_graph
from src.utils.graph_partitioning import partition_graph
from src.utils.logger import get_logger
from src.utils.utils import log_config


def main(save_path="./"):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOGGER = get_logger(
        name=f"{now}_{config.dataset.dataset_name}",
        log_on_file=True,
        save_path=save_path,
    )

    log_config(_LOGGER, config)
    graph = define_graph(config.dataset.dataset_name)
    graph.add_masks(
        train_size=config.subgraph.train_ratio,
        test_size=config.subgraph.test_ratio,
    )

    subgraphs = partition_graph(
        graph, config.subgraph.num_subgraphs, config.subgraph.partitioning
    )
    server = FedPubServer(graph, save_path=save_path, logger=_LOGGER)
    for subgraph in subgraphs:
        server.add_client(subgraph)
    server.start()


if __name__ == "__main__":
    save_path = (
        "./results/"
        f"{config.dataset.dataset_name}/"
        f"{config.subgraph.partitioning}/"
        f"{config.subgraph.num_subgraphs}/all/"
    )
    os.makedirs(save_path, exist_ok=True)
    main(save_path)
