import os
from ast import List

import torch
from tqdm import tqdm
from src.GNN_client import GNNClient

from src.utils.utils import *
from src.utils.graph import Graph
from src.utils.config_parser import Config
from src.server import Server
from src.GNN_classifier import GNNClassifier

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

            self.set_SFV(self.graph.structural_features)

            if propagate_type == "MP":
                abar = self.obtain_a()

                self.share_abar(abar)
                self.set_abar(abar)

                self.share_SFV()

        self.initialized = True

    def obtain_a(self):
        if config.structure_model.estimate:
            abar = estimate_a(
                self.graph.edge_index,
                self.graph.num_nodes,
                config.structure_model.mp_layers,
            )
        else:
            abar = obtain_a(
                self.graph.edge_index,
                self.graph.num_nodes,
                config.structure_model.mp_layers,
            )

        # abar_ = abar.to_dense().numpy()
        # abar1_ = abar1.to_dense().numpy()
        # e = np.mean(np.abs(abar_ - abar1_) ** 2)
        # print(e)

        return abar

    def share_abar(self, abar):
        num_nodes = self.graph.num_nodes
        row, col, val = abar.coo()

        client: GNNClient
        for client in self.clients:
            nodes = client.get_nodes()
            node_map = client.graph.node_map

            cond = torch.isin(row, nodes)
            row_i = row[cond]
            row_i = torch.tensor(itemgetter(*np.array(row_i))(node_map))
            col_i = col[cond]
            val_i = val[cond]
            abar_i = SparseTensor(
                row=row_i,
                col=col_i,
                value=val_i,
                sparse_sizes=(len(nodes), num_nodes),
            )

            client.set_abar(abar_i)

    def create_SFV(self):
        pass

    def share_SFV(self):
        SFV = self.graph.structural_features

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
    ):
        self.initialize_FL(
            propagate_type=propagate_type,
            structure=structure,
            structure_type=structure_type,
        )

        if FL & structure:
            model_type = "SDGA"
        elif FL and not structure:
            model_type = "FLGA GNN"
        elif not FL and structure:
            model_type = "LocaL SDGA"
        else:
            model_type = "Local GNN"

        if propagate_type == "MP":
            model_type += "_MP"

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
    ):
        self.initialize_FL(
            propagate_type=propagate_type,
            structure=structure,
            structure_type=structure_type,
        )

        if FL & structure:
            model_type = "SDWA"
        elif FL and not structure:
            model_type = "FLWA GNN"
        elif not FL and structure:
            model_type = "LocaL SDWA"
        else:
            model_type = "Local GNN"

        if propagate_type == "MP":
            model_type += "_MP"

        return super().joint_train_w(
            epochs=epochs,
            FL=FL,
            log=log,
            plot=plot,
            model_type=model_type,
        )
