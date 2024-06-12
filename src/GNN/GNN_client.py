from torch_sparse import SparseTensor

from src import *
from src.GNN.laplace import SLaplace
from src.client import Client
from src.classifier import Classifier
from src.utils.graph import AGraph, Graph
from src.GNN.fGNN import FGNN
from src.GNN.DGCN import DGCN, SDGCN, SDGCNMaster
from src.GNN.sGNN import SClassifier, SGNNMaster, SGNNSlave
from src.GNN.GNN_classifier import (
    FedDGCN,
    FedDGCNMaster,
    FedGNNMaster,
    FedLaplaceClassifier,
    FedMLPClassifier,
    FedSlave,
)


class GNNClient(Client):
    def __init__(self, graph: Graph, id: int = 0):
        super().__init__(graph=graph, id=id, classifier_type="GNN")
        # LOGGER.info(f"Number of edges: {self.graph.num_edges}")
        self.classifier: Classifier | None = None

    def create_FDGCN_data(self) -> AGraph:
        abar = calc_a(
            self.graph.edge_index,
            self.graph.num_nodes,
            config.feature_model.DGCN_layers,
        )

        graph = AGraph(
            abar=abar,
            edge_index=self.graph.edge_index,
            x=self.graph.x,
            y=self.graph.y,
            node_ids=self.graph.node_ids,
            train_mask=self.graph.train_mask,
            val_mask=self.graph.val_mask,
            test_mask=self.graph.test_mask,
            num_classes=self.graph.num_classes,
        )
        return graph

    def create_SGNN_data(self, **kwargs) -> Graph:
        SFV = kwargs.get("SFV", None)
        SFV_ = torch.tensor(
            SFV.detach().cpu().numpy(),
            requires_grad=SFV.requires_grad,
            device=dev,
        )
        graph = Graph(
            x=SFV_,
            y=self.graph.y,
            edge_index=self.graph.get_edges(),
            node_ids=self.graph.node_ids,
            inter_edges=self.graph.inter_edges,
            external_nodes=self.graph.external_nodes,
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
        smodel_type=config.model.smodel_type,
        fmodel_type=config.model.fmodel_type,
        data_type="feature",
        **kwargs,
    ) -> None:
        self.classifier = None
        if data_type == "feature":
            if fmodel_type == "GNN":
                self.classifier = FGNN(self.graph)
            else:
                graph = self.create_FDGCN_data()
                self.classifier = DGCN(graph)
        elif data_type == "structure":
            if smodel_type == "GNN":
                if self.id == "Server":
                    graph = self.create_SGNN_data(**kwargs)
                    self.classifier = SGNNMaster(graph)
                else:
                    server_embedding_func = kwargs.get("server_embedding_func", None)
                    self.classifier = SGNNSlave(self.graph, server_embedding_func)
            elif smodel_type == "DGCN":
                graph = self.create_SDGCN_data(**kwargs)
                self.classifier = SDGCN(graph)
            elif smodel_type == "CentralDGCN":
                if self.id == "Server":
                    graph = self.create_SDGCN_data(**kwargs)
                    self.classifier = SDGCNMaster(graph)
                else:
                    server_embedding_func = kwargs.get("server_embedding_func", None)
                    self.classifier = SGNNSlave(self.graph, server_embedding_func)
            elif smodel_type == "Laplace":
                sgraph = self.create_SGNN_data(**kwargs)
                self.classifier = SLaplace(sgraph)
            elif smodel_type == "MLP":
                sgraph = self.create_SGNN_data(**kwargs)
                self.classifier = SClassifier(sgraph)

        elif data_type == "f+s":
            if fmodel_type == "GNN":
                fgraph = self.graph
            else:
                fgraph = self.create_FDGCN_data()

            if smodel_type == "GNN":
                if self.id == "Server":
                    sgraph = self.create_SGNN_data(**kwargs)
                    self.classifier = FedGNNMaster(fgraph, sgraph)
                else:
                    server_embedding_func = kwargs.get("server_embedding_func", None)
                    self.classifier = FedSlave(fgraph, server_embedding_func)
            elif smodel_type == "DGCN":
                sgraph = self.create_SDGCN_data(**kwargs)
                self.classifier = FedDGCN(fgraph, sgraph)
            elif smodel_type == "CentralDGCN":
                if self.id == "Server":
                    sgraph = self.create_SDGCN_data(**kwargs)
                    self.classifier = FedDGCNMaster(fgraph, sgraph)
                else:
                    server_embedding_func = kwargs.get("server_embedding_func", None)
                    self.classifier = FedSlave(fgraph, server_embedding_func)
            elif smodel_type == "Laplace":
                sgraph = self.create_SGNN_data(**kwargs)
                self.classifier = FedLaplaceClassifier(fgraph, sgraph)
            elif smodel_type == "MLP":
                sgraph = self.create_SGNN_data(**kwargs)
                self.classifier = FedMLPClassifier(fgraph, sgraph)

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
        smodel_type=config.model.smodel_type,
        fmodel_type=config.model.fmodel_type,
        data_type="feature",
        structure_type=config.structure_model.structure_type,
        log=True,
        plot=True,
        **kwargs,
    ):
        model_type = f"Server {data_type} {smodel_type}-{fmodel_type}"
        self.initialize(
            smodel_type=smodel_type,
            fmodel_type=fmodel_type,
            data_type=data_type,
            structure_type=structure_type,
        )
        return super().train_local_model(
            epochs=epochs,
            log=log,
            plot=plot,
            model_type=model_type,
        )
