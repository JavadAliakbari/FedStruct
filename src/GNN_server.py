from operator import itemgetter
import os
from ast import List

import torch
from src.GNN_client import GNNClient

from src.utils.utils import *
from src.utils.graph import Graph
from src.utils.config_parser import Config
from src.server import Server

dev = os.environ.get("device", "cpu")
device = torch.device(dev)

path = os.environ.get("CONFIG_PATH")
config = Config(path)


class GNNServer(Server, GNNClient):
    def __init__(
        self,
        graph: Graph,
        num_classes,
        save_path="./",
        logger=None,
    ):
        super().__init__(
            graph=graph,
            num_classes=num_classes,
            save_path=save_path,
            logger=logger,
        )

        self.clients: List[GNNClient] = []

    def add_client(self, subgraph):
        client = GNNClient(
            graph=subgraph,
            id=self.num_clients,
            num_classes=self.num_classes,
            save_path=self.save_path,
            logger=self.LOGGER,
        )

        self.clients.append(client)
        self.num_clients += 1

    def initialize_FL(
        self,
        propagate_type=config.model.propagate_type,
        structure=False,
        structure_type=config.structure_model.structure_type,
    ) -> None:
        self.initialize(
            propagate_type=propagate_type,
            structure=structure,
            structure_type=structure_type,
        )
        client: GNNClient
        for client in self.clients:
            client.initialize(
                propagate_type=propagate_type,
                structure=structure,
                structure_type=structure_type,
                get_structure_embeddings=self.get_structure_embeddings2,
            )

        if structure:
            self.graph.add_structural_features(
                structure_type=structure_type,
                num_structural_features=config.structure_model.num_structural_features,
            )

            if propagate_type == "DGCN":
                if self.graph.abar is None:
                    abar = self.obtain_a()
                else:
                    abar = self.graph.abar

                self.share_abar(abar)

            self.share_SFV()

        self.initialized = True

    def obtain_a(self):
        if config.structure_model.estimate:
            abar = estimate_a(
                self.graph.edge_index,
                self.graph.num_nodes,
                config.structure_model.DGCN_layers,
            )
        else:
            abar = obtain_a(
                self.graph.edge_index,
                self.graph.num_nodes,
                config.structure_model.DGCN_layers,
            )

        # abar_ = abar.to_dense().numpy()
        # abar1_ = abar1.to_dense().numpy()
        # e = np.mean(np.abs(abar_ - abar1_) ** 2)
        # print(e)

        return abar

    def share_abar(self, abar):
        if abar is None:
            for client in self.clients:
                client.set_abar(None)

            return

        if dev == "mps":
            local_dev = "cpu"
        else:
            local_dev = dev

        num_nodes = self.graph.num_nodes

        client: GNNClient
        for client in self.clients:
            nodes = client.get_nodes().to(local_dev)
            num_nodes_i = client.num_nodes()
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
                abar_i = abar_i.to_dense().to(device)

            client.set_abar(abar_i)

        self.set_abar(abar)

    def share_SFV(self):
        SFV = self.graph.structural_features

        self.set_SFV(SFV)

        client: GNNClient
        for client in self.clients:
            client.set_SFV(SFV)

    def joint_train_g(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        FL=True,
        structure=False,
        structure_type=config.structure_model.structure_type,
        log=True,
        plot=True,
        model_type="",
    ):
        self.initialize_FL(
            propagate_type=propagate_type,
            structure=structure,
            structure_type=structure_type,
        )

        if FL & structure:
            model_type += "SDGA"
        elif FL and not structure:
            model_type += "FLGA GNN"
        elif not FL and structure:
            model_type += "LocaL SDGA"
        else:
            model_type += "Local GNN"

        if propagate_type == "DGCN":
            model_type += "_DGCN"

        return super().joint_train_g(
            epochs=epochs,
            FL=FL,
            log=log,
            plot=plot,
            model_type=model_type,
        )

    def joint_train_w(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        FL=True,
        structure=False,
        structure_type=config.structure_model.structure_type,
        log=True,
        plot=True,
        model_type="",
    ):
        self.initialize_FL(
            propagate_type=propagate_type,
            structure=structure,
            structure_type=structure_type,
        )

        if FL & structure:
            model_type += "SDWA"
        elif FL and not structure:
            model_type += "FLWA GNN"
        elif not FL and structure:
            model_type += "LocaL SDWA"
        else:
            model_type += "Local GNN"

        if propagate_type == "DGCN":
            model_type += "_DGCN"

        return super().joint_train_w(
            epochs=epochs,
            FL=FL,
            log=log,
            plot=plot,
            model_type=model_type,
        )
