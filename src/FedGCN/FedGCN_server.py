from ast import List

from src.GNN.GNN_client import GNNClient

from src import *
from src.utils.graph import Graph
from src.server import Server


class FedGCNServer(Server, GNNClient):
    def __init__(self, graph: Graph):
        super().__init__(graph=graph)

        self.clients: List[GNNClient] = []

    def add_client(self, subgraph):
        client = GNNClient(
            graph=subgraph,
            id=self.num_clients,
        )

        self.clients.append(client)
        self.num_clients += 1

    def initialize(self, **kwargs) -> None:
        super().initialize(
            data_type="feature",
            fmodel_type="GNN",
            **kwargs,
        )

    def initialize_FL(self, **kwargs) -> None:
        self.initialize(**kwargs)
        client: GNNClient
        for client in self.clients:
            client.initialize(
                data_type="feature",
                fmodel_type="GNN",
                **kwargs,
            )

    def joint_train_g(
        self,
        epochs=config.model.iterations,
        log=True,
        plot=True,
        **kwargs,
    ):
        self.initialize_FL(**kwargs)
        model_type = f"FL FedGCN GA"

        return super().joint_train_g(
            epochs=epochs,
            FL=True,
            log=log,
            plot=plot,
            model_type=model_type,
        )

    def joint_train_w(
        self,
        epochs=config.model.iterations,
        log=True,
        plot=True,
        **kwargs,
    ):
        self.initialize_FL(**kwargs)
        model_type = f"FL FedGCN WA"

        return super().joint_train_w(
            epochs=epochs,
            FL=True,
            log=log,
            plot=plot,
            model_type=model_type,
        )
