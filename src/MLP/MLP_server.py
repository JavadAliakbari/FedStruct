import os
from ast import List

from src.utils.data import Data
from src.utils.config_parser import Config
from src.server import Server
from src.MLP.MLP_client import MLPClient

path = os.environ.get("CONFIG_PATH")
config = Config(path)


class MLPServer(Server, MLPClient):
    def __init__(
        self,
        graph: Data,
        save_path="./",
        logger=None,
    ):
        super().__init__(
            graph=graph,
            save_path=save_path,
            logger=logger,
        )
        self.clients: List[MLPClient] = []

    def add_client(self, subgraph):
        client = MLPClient(
            graph=subgraph,
            id=self.num_clients,
            save_path=self.save_path,
            logger=self.LOGGER,
        )

        self.clients.append(client)
        self.num_clients += 1

    def initialize_FL(self) -> None:
        self.initialize()

        client: MLPClient
        for client in self.clients:
            client.initialize()

        self.initialized = True

    def joint_train_g(
        self,
        epochs=config.model.iterations,
        log=True,
        plot=True,
        FL=True,
    ):
        self.initialize_FL()

        if FL:
            model_type = "FLGA MLP"
        else:
            model_type = "Local MLP"

        return super().joint_train_g(
            epochs=epochs, log=log, plot=plot, FL=FL, model_type=model_type
        )

    def joint_train_w(
        self,
        epochs=config.model.iterations,
        log=True,
        plot=True,
        FL=True,
    ):
        self.initialize_FL()

        if FL:
            model_type = "FLWA MLP"
        else:
            model_type = "Local MLP"

        return super().joint_train_w(
            epochs=epochs, log=log, plot=plot, FL=FL, model_type=model_type
        )
