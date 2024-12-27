from ast import List

from src import *
from src.utils.graph import Graph
from src.GNN.GNN_server import GNNServer
from src.fedsage.fedsage_client import FedSAGEClient


class FedSAGEServer(GNNServer, FedSAGEClient):
    def __init__(self, graph: Graph):
        super().__init__(graph=graph)

        self.clients: List[FedSAGEClient] = []

    def add_client(self, subgraph):
        client = FedSAGEClient(graph=subgraph, id=self.num_clients)

        self.clients.append(client)
        self.num_clients += 1

    def initialize_FL(self, **kwargs) -> None:
        self.initialize()
        client: FedSAGEClient
        for client in self.clients:
            client.initialize()

    def initialize_neighgens(self) -> None:
        # self.initialize_neighgen()
        client: FedSAGEClient
        for client in self.clients:
            client.initialize_neighgen()

    def create_mend_graphs(self, predict=True):
        client: FedSAGEClient
        for client in self.clients:
            client.create_mend_graph(predict=predict)

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

    def train_neighgen_clients(self, inter_client_features_creators=[], predict=True):
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
                inter_client_features_creators_client, predict=predict
            )
            results.append(result)

        return results

    def test_neighgen_models(self):
        results = []

        client: FedSAGEClient
        for client in self.clients:
            result = client.get_neighgen_test_results()
            results.append(result)

        return results

    def joint_train_neighgen(
        self,
        epochs=config.fedsage.neighgen_epochs,
        inter_client_features_creators=[],
        predict=True,
        log=True,
        plot=True,
    ):
        if log:
            LOGGER.info(f"Neighgen starts!")
            bar = tqdm(total=epochs, position=0)

        coef = [client.num_nodes() / self.num_nodes() for client in self.clients]

        average_results = []
        for epoch in range(epochs):
            self.reset_neighgen_trainings()

            self.set_neighgen_train_mode()
            results = self.train_neighgen_clients(
                inter_client_features_creators,
                predict=predict,
            )
            average_result = sum_lod(results, coef)
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
            plot_path = f"{save_path}/plots/{now}/"
            plot_metrics(average_results, title=title, save_path=plot_path)

        test_results = self.test_neighgen_models()
        average_result = sum_lod(test_results, coef)
        final_results = {}
        for cleint, test_result in zip(self.clients, test_results):
            final_results[f"Client{cleint.id}"] = test_result
        final_results["Average"] = average_result
        if log:
            self.report_test_results(final_results)

        return final_results

    def train_locsages(
        self,
        epochs=config.model.iterations,
        smodel_type=config.model.smodel_type,
        fmodel_type=config.model.fmodel_type,
        log=True,
        plot=True,
    ):
        if log:
            LOGGER.info("Locsage+ starts!")
        self.initialize_neighgens()
        self.joint_train_neighgen(
            epochs=config.fedsage.neighgen_epochs,
            log=log,
            plot=plot,
        )
        self.create_mend_graphs()
        res1 = self.joint_train_w(
            epochs=epochs,
            smodel_type=smodel_type,
            fmodel_type=fmodel_type,
            log=log,
            plot=plot,
            model_type="Neighgen",
        )

        res2 = self.joint_train_g(
            epochs=epochs,
            smodel_type=smodel_type,
            fmodel_type=fmodel_type,
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
        predict=True,
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
                    if predict:
                        inter_client_features_creators_client.append(
                            other_client.create_inter_features
                        )
                    else:
                        inter_client_features_creators_client.append(
                            other_client.create_inter_features2
                        )

            inter_client_features_creators.append(inter_client_features_creators_client)

        self.joint_train_neighgen(
            epochs=epochs,
            inter_client_features_creators=inter_client_features_creators,
            predict=predict,
            log=log,
            plot=plot,
        )

    def train_fedSage_plus(
        self,
        epochs=config.model.iterations,
        model="both",
        predict=True,
        log=True,
        plot=True,
    ):
        if log:
            LOGGER.info("FedSage+ starts!")

        self.initialize_neighgens()
        self.train_fedgen(predict=predict, log=log, plot=plot)
        self.create_mend_graphs(predict=predict)
        results = {}
        if model == "WA" or model == "both":
            res1 = self.joint_train_w(
                epochs=epochs,
                fmodel_type="GNN",
                data_type="feature",
                log=log,
                plot=plot,
                model_type="Neighgen",
            )
            results["WA"] = res1

        if model == "GA" or model == "both":
            res2 = self.joint_train_g(
                epochs=epochs,
                fmodel_type="GNN",
                data_type="feature",
                log=log,
                plot=plot,
                model_type="Neighgen",
            )
            results["GA"] = res2

        return results
