import os
from tqdm import tqdm

from src.utils.utils import *
from src.client import Client
from src.utils.graph import Graph
from src.utils.config_parser import Config

path = os.environ.get("CONFIG_PATH")
config = Config(path)


class Server(Client):
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
            id="Server",
            save_path=save_path,
            logger=logger,
        )
        self.clients = None
        self.num_clients = 0

        self.LOGGER.info(f"Number of features: {self.graph.num_features}")

    def remove_clients(self):
        self.clients.clear()
        self.num_clients = 0

    def share_weights(self):
        server_weights = self.state_dict()

        client: Client
        for client in self.clients:
            client.load_state_dict(server_weights)

    def report_results(self, results, framework=""):
        client: Client
        for client, result in zip(self.clients, results):
            client.report_result(result, framework)

    def share_grads(self, grads):
        client: Client
        for client in self.clients:
            client.set_grads(grads)

        self.set_grads(grads)

    def set_train_mode(self, mode: bool = True):
        self.train(mode)

        client: Client
        for client in self.clients:
            client.train(mode)

    def train_clients(self, scale=False):
        results = []

        client: Client
        for client in self.clients:
            result = client.get_train_results(scale)
            results.append(result)

        return results

    def test_clients(self):
        results = {}

        client: Client
        for client in self.clients:
            result = client.get_test_results()
            results[f"Client{client.id}"] = result

        return results

    def report_test_results(self, test_results):
        for client_id, result in test_results.items():
            for key, val in result.items():
                self.LOGGER.info(f"{client_id} {key}: {val:0.4f}")

    def report_average_results(self, average_result):
        sum = sum_dictofdicts(average_result)
        for key, val in sum.items():
            self.LOGGER.info(f"Average {key}: {val / len(average_result):0.4f}")

    def report_server_test(self):
        test_acc = self.test_classifier()
        self.LOGGER.info(f"Server test: {test_acc:0.4f}")

    def update_models(self):
        client: Client
        for client in self.clients:
            client.update_model()

        self.update_model()

    def reset_trainings(self):
        self.reset_model()
        client: Client
        for client in self.clients:
            client.reset_model()

    def joint_train_g(
        self,
        epochs=config.model.epoch_classifier,
        FL=True,
        log=True,
        plot=True,
        model_type="GNN",
    ):
        self.LOGGER.info(f"{model_type} starts!")

        if FL:
            self.share_weights()

        if log:
            bar = tqdm(total=epochs, position=0)

        average_results = []
        for epoch in range(epochs):
            self.reset_trainings()

            self.set_train_mode()
            results = self.train_clients(True & FL)
            average_result = calc_average_result(results)
            average_result["Epoch"] = epoch + 1
            average_results.append(average_result)

            if FL:
                clients_grads = get_grads(self.clients)
                grads = sum_grads(clients_grads, self.num_nodes())
                self.share_grads(grads)

            self.update_models()

            if log:
                bar.set_postfix(average_result)
                bar.update()

                if epoch == epochs - 1:
                    self.report_results(results, "Joint Training")

        if plot:
            title = f"Average joint Training {model_type}"
            plot_metrics(average_results, title=title, save_path=self.save_path)

        if log:
            self.report_server_test()
        test_results = self.test_clients()
        average_result = calc_average_result2(test_results)
        test_results["Average"] = average_result
        if log:
            self.report_test_results(test_results)

        return test_results

    def joint_train_w(
        self,
        epochs=config.model.epoch_classifier,
        log=True,
        plot=True,
        FL=True,
        model_type="GNN",
    ):
        self.LOGGER.info(f"{model_type} starts!")

        if log:
            bar = tqdm(total=epochs, position=0)

        average_results = []
        for epoch in range(epochs):
            if FL:
                self.share_weights()
            self.reset_trainings()

            self.set_train_mode()
            results = self.train_clients()
            average_result = calc_average_result(results)
            average_result["Epoch"] = epoch + 1
            average_results.append(average_result)

            if FL:
                clients_grads = get_grads(self.clients, True)
                grads = sum_grads(clients_grads)
                self.share_grads(grads)

            self.update_models()

            if FL:
                mean_weights = sum_weights(self.clients)
                self.load_state_dict(mean_weights)

            if log:
                bar.set_postfix(average_result)
                bar.update()

                if epoch == epochs - 1:
                    self.report_results(results, "Joint Training")

        if plot:
            title = f"Average Joint Training {model_type}"
            plot_metrics(average_results, title=title, save_path=self.save_path)

        if log:
            self.report_server_test()
        test_results = self.test_clients()
        average_result = calc_average_result2(test_results)
        test_results["Average"] = average_result
        if log:
            self.report_test_results(test_results)

        return test_results
