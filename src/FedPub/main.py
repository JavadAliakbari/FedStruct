from parser_file import Parser
from datetime import datetime

from src.FedPub.server import Server
from src.FedPub.utils import *
from src.utils.config_parser import Config
from src.utils.define_graph import define_graph
from src.utils.graph_partitioning import partition_graph

path = os.environ.get("CONFIG_PATH")
config = Config(path)


def main():
    graph = define_graph(config.dataset.dataset_name)
    graph.add_masks(
        train_size=config.subgraph.train_ratio,
        test_size=config.subgraph.test_ratio,
    )

    subgraphs = partition_graph(
        graph, config.subgraph.num_subgraphs, config.subgraph.partitioning
    )
    args = set_config()
    server = Server(graph, args)
    for subgraph in subgraphs:
        server.add_client(subgraph)
    server.start()


def set_config():
    args = Parser().parse()
    args.model = "fedpub"
    args.dataset = "Cora"
    args.frac = 1.0
    args.n_rnds = config.model.epoch_classifier
    args.n_eps = 1
    args.n_clients = config.subgraph.num_subgraphs
    args.clsf_mask_one = True
    args.laye_mask_one = True
    args.norm_scale = 3
    args.seed = 42

    args.base_lr = 0.01
    args.min_lr = 1e-3
    args.momentum_opt = 0.9
    args.weight_decay = 1e-6
    args.warmup_epochs = 10
    args.base_momentum = 0.99
    args.final_momentum = 1.0

    if args.dataset == "Cora":
        args.n_feat = 1433
        args.n_clss = 7
        args.n_clients = 10 if args.n_clients == None else args.n_clients
        args.base_lr = 0.01 if args.lr == None else args.lr
    if args.dataset == "PubMed":
        args.n_feat = 500
        args.n_clss = 3
        args.n_clients = 10 if args.n_clients == None else args.n_clients
        args.base_lr = 0.01 if args.lr == None else args.lr

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial = f"{args.dataset}_{args.mode}/clients_{args.n_clients}/{now}_{args.model}"

    args.data_path = f"{args.base_path}/datasets"
    args.checkpt_path = f"{args.base_path}/checkpoints/{trial}"
    args.log_path = f"{args.base_path}/logs/{trial}"

    if args.debug == True:
        args.checkpt_path = f"{args.base_path}/debug/checkpoints/{trial}"
        args.log_path = f"{args.base_path}/debug/logs/{trial}"

    return args


if __name__ == "__main__":
    main()
