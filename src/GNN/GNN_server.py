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
        smodel_type=config.model.smodel_type,
        fmodel_type=config.model.fmodel_type,
        data_type="feature",
        **kwargs,
    ) -> None:
        share = {}
        if data_type in ["structure", "f+s"]:
            structure_type = kwargs.get(
                "structure_type", config.structure_model.structure_type
            )
            num_structural_features = kwargs.get(
                "num_structural_features",
                config.structure_model.num_structural_features,
            )
            if smodel_type in ["SpectralLaplace", "LanczosLaplace"]:
                num_spectral_features = config.spectral.spectral_len
            else:
                num_spectral_features = None

            self.graph.add_structural_features(
                structure_type=structure_type,
                num_structural_features=num_structural_features,
                num_spectral_features=num_spectral_features,
            )
            SFV = self.graph.structural_features
            share["SFV"] = SFV

            if smodel_type in ["DGCN", "CentralDGCN"]:
                self.graph.obtain_a()
                share["abar"] = self.graph.abar

            if smodel_type == "CentralDGCN":
                share["server_embedding_func"] = self.classifier.get_embeddings_func()

            if smodel_type == "SpectralLaplace":
                self.graph.create_L()
                self.graph.calc_eignvalues(estimate=False)
                share["U"] = self.graph.U
                share["D"] = self.graph.D
            if smodel_type == "LanczosLaplace":
                self.graph.create_L()
                self.graph.calc_eignvalues(estimate=True)
                share["U"] = self.graph.U
                share["D"] = self.graph.D

        kwargs.update(share)

        super().initialize(
            smodel_type=smodel_type,
            fmodel_type=fmodel_type,
            data_type=data_type,
            **kwargs,
        )

        return share

    def initialize_FL(
        self,
        smodel_type=config.model.smodel_type,
        fmodel_type=config.model.fmodel_type,
        data_type="feature",
        **kwargs,
    ) -> None:
        share = self.initialize(
            smodel_type=smodel_type,
            fmodel_type=fmodel_type,
            data_type=data_type,
            **kwargs,
        )

        kwargs.update(share)

        client: GNNClient
        for client in self.clients:
            client.initialize(
                smodel_type=smodel_type,
                fmodel_type=fmodel_type,
                data_type=data_type,
                **kwargs,
            )

    def joint_train_g(
        self,
        epochs=config.model.iterations,
        smodel_type=config.model.smodel_type,
        fmodel_type=config.model.fmodel_type,
        FL=True,
        data_type="feature",
        log=True,
        plot=True,
        model_type="",
        **kwargs,
    ):
        self.initialize_FL(
            smodel_type=smodel_type,
            fmodel_type=fmodel_type,
            data_type=data_type,
            **kwargs,
        )
        if FL:
            model_type = f"FL {data_type} {smodel_type}-{fmodel_type} GA"
        else:
            model_type = f"Local {data_type} {smodel_type}-{fmodel_type} GA"

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
        smodel_type=config.model.smodel_type,
        fmodel_type=config.model.fmodel_type,
        FL=True,
        data_type="feature",
        log=True,
        plot=True,
        model_type="",
        **kwargs,
    ):
        self.initialize_FL(
            smodel_type=smodel_type,
            fmodel_type=fmodel_type,
            data_type=data_type,
            **kwargs,
        )
        if FL:
            model_type = f"FL {data_type} {smodel_type}-{fmodel_type} WA"
        else:
            model_type = f"Local {data_type} {smodel_type}-{fmodel_type} WA"

        return super().joint_train_w(
            epochs=epochs,
            FL=FL,
            log=log,
            plot=plot,
            model_type=model_type,
        )
