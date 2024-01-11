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

    def initialize_neighgens(self) -> None:
        # self.initialize_neighgen()
        client: FedSAGEClient
        for client in self.clients:
            client.initialize_neighgen()

    def create_mend_graphs(self):
        client: FedSAGEClient
        for client in self.clients:
            client.create_mend_graph()

    def reset_neighgen_trainings(self):
        # self.reset_neighgen_model()
        client: FedSAGEClient
        for client in self.clients:
            client.reset_neighgen_model()

    def update_neighgen_models(self):
        client: FedSAGEClient
        for client in self.clients:
            client.update_neighgen_model()

        # self.update_neighgen_model()

    def set_neighgen_train_mode(self, mode: bool = True):
        # self.neighgen_train_mode(mode)

        client: FedSAGEClient
        for client in self.clients:
            client.neighgen_train_mode(mode)

    def train_neighgen_clients(self, inter_client_features_creators=[]):
        results = []

        client: FedSAGEClient
        for idx, client in enumerate(self.clients):
            if len(inter_client_features_creators) == 0:
                inter_client_features_creators_client = []
            else:
                inter_client_features_creators_client = inter_client_features_creators[
                    idx
                ]
            result = client.get_neighgen_train_results(
                inter_client_features_creators_client
            )
            results.append(result)

        return results

    def test_neighgen_models(self):
        results = {}

        client: FedSAGEClient
        for client in self.clients:
            result = client.get_neighgen_test_results()
            results[f"Client{client.id}"] = result

        return results

    def joint_train_neighgen(
        self,
        epochs=config.fedsage.neighgen_epochs,
        inter_client_features_creators=[],
        log=True,
        plot=True,
    ):
        self.LOGGER.info(f"Neighgen starts!")

        if log:
            bar = tqdm(total=epochs, position=0)

        average_results = []
        for epoch in range(epochs):
            self.reset_neighgen_trainings()

            self.set_neighgen_train_mode()
            results = self.train_neighgen_clients(inter_client_features_creators)
            average_result = calc_average_result(results)
            average_result["Epoch"] = epoch + 1
            average_results.append(average_result)

            self.update_neighgen_models()

            if log:
                bar.set_postfix(average_result)
                bar.update()

                if epoch == epochs - 1:
                    self.report_results(results, "Joint Training")

        if plot:
            title = f"Average joint Training Neighgen"
            plot_metrics(average_results, title=title, save_path=self.save_path)

        test_results = self.test_neighgen_models()
        average_result = calc_average_result2(test_results)
        test_results["Average"] = average_result
        if log:
            self.report_test_results(test_results)

        return test_results

    def train_locsages(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        log=True,
        plot=True,
    ):
        self.LOGGER.info("Locsage+ starts!")
        self.initialize_neighgens()
        self.joint_train_neighgen(
            epochs=config.fedsage.neighgen_epochs,
            log=log,
            plot=plot,
        )
        self.create_mend_graphs()
        res1 = self.joint_train_w(
            epochs=epochs,
            propagate_type=propagate_type,
            log=log,
            plot=plot,
            model_type="Neighgen",
        )

        res2 = self.joint_train_g(
            epochs=epochs,
            propagate_type=propagate_type,
            log=log,
            plot=plot,
            model_type="Neighgen",
        )

        results = {
            "WA": res1,
            "GA": res2,
        }

        return results

    def train_fedgen(
        self,
        epochs=config.fedsage.neighgen_epochs,
        log=True,
        plot=True,
    ):
        client: FedSAGEClient
        other_client: FedSAGEClient

        inter_client_features_creators = []
        for client in self.clients:
            inter_client_features_creators_client = []
            for other_client in self.clients:
                if other_client.id != client.id:
                    inter_client_features_creators_client.append(
                        other_client.create_inter_features
                    )

            inter_client_features_creators.append(inter_client_features_creators_client)

        self.joint_train_neighgen(
            epochs=epochs,
            inter_client_features_creators=inter_client_features_creators,
            log=log,
            plot=plot,
        )

    def train_fedSage_plus(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        model="both",
        log=True,
        plot=True,
    ):
        self.LOGGER.info("FedSage+ starts!")

        self.initialize_neighgens()
        self.train_fedgen(log=log, plot=plot)
        self.create_mend_graphs()
        results = {}
        if model == "WA" or model == "both":
            res1 = self.joint_train_w(
                epochs=epochs,
                propagate_type=propagate_type,
                log=log,
                plot=plot,
                model_type="Neighgen",
            )
            results["WA"] = res1

        if model == "GA" or model == "both":
            res2 = self.joint_train_g(
                epochs=epochs,
                propagate_type=propagate_type,
                log=log,
                plot=plot,
                model_type="Neighgen",
            )
            results["GA"] = res2

        return results
