import os
from ast import List

from src.utils.utils import *
from src.utils.graph import Graph
from src.utils.config_parser import Config
from src.GNN_server import GNNServer
from src.fedsage_client import FedSAGEClient

path = os.environ.get("CONFIG_PATH")
config = Config(path)


class FedSAGEServer(GNNServer, FedSAGEClient):
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

        self.clients: List[FedSAGEClient] = []

    def add_client(self, subgraph):
        client = FedSAGEClient(
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
        **kwargs,
    ) -> None:
        self.initialize(propagate_type)
        client: FedSAGEClient
        for client in self.clients:
            client.initialize(propagate_type)

    def train_neighgens(self) -> None:
        client: FedSAGEClient
        for client in self.clients:
            client.train_neighgen()

    def create_mend_graphs(self):
        client: FedSAGEClient
        for client in self.clients:
            client.create_mend_graph()

    def train_locsages(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        log=True,
        plot=True,
    ):
        self.LOGGER.info("Locsage+ starts!")
        self.train_neighgens()
        self.create_mend_graphs()
        self.joint_train_w(
            epochs=epochs,
            propagate_type=propagate_type,
            log=log,
            plot=plot,
            FL=False,
        )

    def train_fedgen(self, log=True, plot=True):
        client: FedSAGEClient
        other_client: FedSAGEClient
        for client in self.clients:
            inter_client_features_creators = []
            for other_client in self.clients:
                if other_client.id != client.id:
                    inter_client_features_creators.append(
                        other_client.create_inter_features
                    )

            client.train_neighgen(
                inter_client_features_creators,
                log=log,
                plot=plot,
            )

    def train_fedSage_plus(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        log=True,
        plot=True,
    ):
        self.LOGGER.info("FedSage+ starts!")
        self.train_fedgen(log=log, plot=plot)
        self.create_mend_graphs()
        res1 = self.joint_train_w(
            epochs=epochs,
            propagate_type=propagate_type,
            log=log,
            plot=plot,
        )

        res2 = self.joint_train_g(
            epochs=epochs,
            propagate_type=propagate_type,
            log=log,
            plot=plot,
        )

        results = {
            "WA": res1,
            "GA": res2,
        }

        return results
