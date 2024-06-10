from ast import List

from src.GNN.GNN_client import GNNClient

from src import *
from src.utils.graph import Graph
from src.server import Server


class GNNServer(Server, GNNClient):
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

    def initialize(
        self,
        propagate_type=config.model.propagate_type,
        data_type="feature",
        **kwargs,
    ) -> None:
        if data_type in ["structure", "f+s"]:
            if propagate_type == "DGCN":
                self.graph.obtain_a()

        SFV = None
        if data_type in ["structure", "f+s"]:
            structure_type = kwargs.get(
                "structure_type", config.structure_model.structure_type
            )
            num_structural_features = kwargs.get(
                "num_structural_features",
                config.structure_model.num_structural_features,
            )
            self.graph.add_structural_features(
                structure_type=structure_type,
                num_structural_features=num_structural_features,
            )
            SFV = self.graph.structural_features

        super().initialize(
            propagate_type=propagate_type,
            data_type=data_type,
            SFV=SFV,
            abar=self.graph.abar,
            edge_index=self.graph.edge_index,
            **kwargs,
        )

    def initialize_FL(
        self,
        propagate_type=config.model.propagate_type,
        data_type="feature",
        **kwargs,
    ) -> None:
        self.initialize(
            propagate_type=propagate_type,
            data_type=data_type,
            **kwargs,
        )
        client: GNNClient
        for client in self.clients:
            client.initialize(
                propagate_type=propagate_type,
                data_type=data_type,
                abar=self.graph.abar,
                SFV=self.graph.structural_features,
                server_embedding_func=self.classifier.get_embeddings_func(),
                edge_index=self.graph.edge_index,
                **kwargs,
            )

    def joint_train_g(
        self,
        epochs=config.model.iterations,
        propagate_type=config.model.propagate_type,
        FL=True,
        data_type="feature",
        log=True,
        plot=True,
        model_type="",
        **kwargs,
    ):
        self.initialize_FL(
            propagate_type=propagate_type,
            data_type=data_type,
            **kwargs,
        )
        if FL:
            model_type = f"FL {data_type} {propagate_type} GA"
        else:
            model_type = f"Local {data_type} {propagate_type} GA"

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
        propagate_type=config.model.propagate_type,
        FL=True,
        data_type="feature",
        log=True,
        plot=True,
        model_type="",
        **kwargs,
    ):
        self.initialize_FL(
            propagate_type=propagate_type,
            data_type=data_type,
            **kwargs,
        )
        if FL:
            model_type = f"FL {data_type} {propagate_type} WA"
        else:
            model_type = f"Local {data_type} {propagate_type} WA"

        return super().joint_train_w(
            epochs=epochs,
            FL=FL,
            log=log,
            plot=plot,
            model_type=model_type,
        )
