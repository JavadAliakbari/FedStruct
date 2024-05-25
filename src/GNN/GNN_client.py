import os

from torch_sparse import SparseTensor

from src.GNN.DGCN import DGCN, SDGCN
from src.GNN.fGNN import FGNN
from src.GNN.sGNN import SGNNMaster, SGNNSlave
from src.classifier import Classifier
from src.utils.utils import *
from src.utils.config_parser import Config
from src.client import Client
from src.utils.graph import AGraph, Graph
from src.GNN.GNN_classifier import FedDGCN, FedGNNMaster, FedGNNSlave

path = os.environ.get("CONFIG_PATH")
config = Config(path)

dev = os.environ.get("device", "cpu")
if dev == "mps":
    local_dev = "cpu"
else:
    local_dev = dev


class GNNClient(Client):
    def __init__(
        self,
        graph: Graph,
        id: int = 0,
        save_path="./",
        logger=None,
    ):
        super().__init__(
            graph=graph,
            id=id,
            classifier_type="GNN",
            save_path=save_path,
            logger=logger,
        )
        # self.LOGGER.info(f"Number of edges: {self.graph.num_edges}")
        self.classifier: Classifier | None = None

    def create_FDGCN_data(self) -> AGraph:
        if self.id == "Server":
            abar = self.graph.abar
        else:
            abar = calc_a(
                self.graph.edge_index,
                self.graph.num_nodes,
                config.feature_model.DGCN_layers,
            )

        graph = AGraph(
            abar=abar,
            x=self.graph.x,
            y=self.graph.y,
            node_ids=self.graph.node_ids,
            train_mask=self.graph.train_mask,
            val_mask=self.graph.val_mask,
            test_mask=self.graph.test_mask,
            num_classes=self.graph.num_classes,
        )
        return graph

    def create_SGNN_master_data(self, **kwargs) -> Graph:
        SFV = kwargs.get("SFV", None)
        SFV_ = torch.tensor(
            SFV.detach().cpu().numpy(),
            requires_grad=SFV.requires_grad,
            device=dev,
        )
        graph = Graph(
            edge_index=self.graph.edge_index,
            x=SFV_,
            y=self.graph.y,
            train_mask=self.graph.train_mask,
            val_mask=self.graph.val_mask,
            test_mask=self.graph.test_mask,
            num_classes=self.graph.num_classes,
        )
        return graph

    def create_SDGCN_data(self, **kwargs) -> AGraph:
        abar = kwargs.get("abar", None)
        abar_i = self.split_abar(abar)

        SFV = kwargs.get("SFV", None)
        SFV_ = torch.tensor(
            SFV.detach().cpu().numpy(),
            requires_grad=SFV.requires_grad,
            device=dev,
        )
        graph = AGraph(
            abar=abar_i,
            x=SFV_,
            y=self.graph.y,
            node_ids=self.graph.node_ids,
            train_mask=self.graph.train_mask,
            val_mask=self.graph.val_mask,
            test_mask=self.graph.test_mask,
            num_classes=self.graph.num_classes,
        )
        return graph

    def initialize(
        self,
        propagate_type=config.model.propagate_type,
        data_type="feature",
        **kwargs,
    ) -> None:
        if data_type == "feature":
            if propagate_type == "GNN":
                self.classifier = FGNN(self.graph)
            if propagate_type == "DGCN":
                graph = self.create_FDGCN_data()
                self.classifier = DGCN(graph)
        elif data_type == "structure":
            if propagate_type == "GNN":
                if self.id == "Server":
                    graph = self.create_SGNN_master_data(**kwargs)
                    self.classifier = SGNNMaster(graph)
                else:
                    server_embedding_func = kwargs.get("server_embedding_func", None)
                    self.classifier = SGNNSlave(self.graph, server_embedding_func)
            elif propagate_type == "DGCN":
                graph = self.create_SDGCN_data(**kwargs)
                self.classifier = SDGCN(graph)
        elif data_type == "f+s":
            if propagate_type == "GNN":
                if self.id == "Server":
                    sgraph = self.create_SGNN_master_data(**kwargs)
                    self.classifier = FedGNNMaster(self.graph, sgraph)
                else:
                    server_embedding_func = kwargs.get("server_embedding_func", None)
                    self.classifier = FedGNNSlave(self.graph, server_embedding_func)
            elif propagate_type == "DGCN":
                fgraph = self.create_FDGCN_data()
                sgraph = self.create_SDGCN_data(**kwargs)
                self.classifier = FedDGCN(fgraph, sgraph)

        self.classifier.create_optimizer()

    def split_abar(self, abar: SparseTensor):
        num_nodes = abar.size()[0]
        nodes = self.get_nodes().to(local_dev)
        num_nodes_i = self.num_nodes()
        indices = torch.arange(num_nodes_i, dtype=torch.long, device=local_dev)
        vals = torch.ones(num_nodes_i, dtype=torch.float32, device=local_dev)
        P = torch.sparse_coo_tensor(
            torch.vstack([indices, nodes]),
            vals,
            (num_nodes_i, num_nodes),
            device=local_dev,
        )
        abar_i = torch.matmul(P, abar)
        if dev != "cuda:0":
            abar_i = abar_i.to_dense().to(dev)
        return abar_i

    def train_local_model(
        self,
        epochs=config.model.iterations,
        propagate_type=config.model.propagate_type,
        data_type="feature",
        structure_type=config.structure_model.structure_type,
        log=True,
        plot=True,
        **kwargs,
    ):
        model_type = f"Server {data_type} {propagate_type}"
        self.initialize(
            propagate_type=propagate_type,
            data_type=data_type,
            structure_type=structure_type,
        )
        return super().train_local_model(
            epochs=epochs,
            log=log,
            plot=plot,
            model_type=model_type,
        )
