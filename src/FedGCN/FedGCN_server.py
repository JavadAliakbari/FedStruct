from ast import List

from src.FedGCN.FedGCN_client import FedGCNClient

from src import *
from src.utils.graph import Graph
from src.server import Server


class FedGCNServer(Server, FedGCNClient):
    def __init__(self, graph: Graph):
        super().__init__(graph=graph)

        self.clients: List[FedGCNClient] = []

    def add_client(self, subgraph):
        client = FedGCNClient(
            graph=subgraph,
            id=self.num_clients,
        )

        self.clients.append(client)
        self.num_clients += 1

    def initialize(self, **kwargs) -> None:
        super().initialize(**kwargs)

    def initialize_FL(
        self,
        **kwargs,
    ) -> None:
        self.initialize(**kwargs)
        client: FedGCNClient
        for client in self.clients:
            client.initialize(**kwargs)

    def joint_train_g(
        self,
        epochs=config.model.iterations,
        FL=True,
        log=True,
        plot=True,
        **kwargs,
    ):
        self.initialize_FL(**kwargs)
        if FL:
            model_type = f"FL FedGCN GA"
        else:
            model_type = f"Local FedGCN GA"

        return super().joint_train_g(
            epochs=epochs,
            FL=FL,
            log=log,
            plot=plot,
            model_type=model_type,
        )

    def joint_train_w(
        self,
        epochs=config.model.iterations,
        FL=True,
        log=True,
        plot=True,
        **kwargs,
    ):
        self.initialize_FL(**kwargs)
        if FL:
            model_type = f"FL FedGCN WA"
        else:
            model_type = f"Local FedGCN WA"

        return super().joint_train_w(
            epochs=epochs,
            FL=FL,
            log=log,
            plot=plot,
            model_type=model_type,
        )
