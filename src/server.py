from tqdm import tqdm

from src import *
from src.client import Client
from src.utils.graph import Graph


class Server(Client):
    def __init__(self, graph: Graph):
        super().__init__(graph=graph, id="Server")
        self.clients = None
        self.num_clients = 0

        # LOGGER.info(f"Number of features: {self.graph.num_features}")

    def remove_clients(self):
        self.clients.clear()
        self.num_clients = 0

    def share_weights(self):
        server_weights = self.state_dict()

        client: Client
        for client in self.clients:
            client.load_state_dict(server_weights)

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

    def train_clients(self, eval_=True):
        results = []

        client: Client
        for client in self.clients:
            result = client.get_train_results(eval_=eval_)
            results.append(result)

        return results

    def test_clients(self):
        results = []
        client: Client
        for client in self.clients:
            result = client.get_test_results()
            results.append(result)

        return results

    def report_results(self, results, framework=""):
        client: Client
        for client, result in zip(self.clients, results):
            client.report_result(result, framework)

    def report_test_results(self, test_results):
        for client_id, result in test_results.items():
            for key, val in result.items():
                LOGGER.info(f"{client_id} {key}: {val:0.4f}")

    def report_server_test(self):
        res = self.test_classifier()
        LOGGER.info(f"Server test: {res[0]:0.4f}")

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
        epochs=config.model.iterations,
        FL=True,
        log=True,
        plot=True,
        model_type="GNN",
    ):
        if log:
            LOGGER.info(f"{model_type} starts!")

        if FL:
            self.share_weights()

        if log:
            bar = tqdm(total=epochs, position=0)

        num_nodes = sum([client.num_nodes() for client in self.clients])
        coef = [client.num_nodes() / num_nodes for client in self.clients]
        average_results = []
        for epoch in range(epochs):
            self.reset_trainings()

            self.set_train_mode()
            results = self.train_clients(eval_=log)
            average_result = sum_lod(results, coef)
            average_result["Epoch"] = epoch + 1
            average_results.append(average_result)

            if FL:
                clients_grads = get_grads(self.clients)
                grads = sum_lod(clients_grads, coef)
                self.share_grads(grads)

            self.update_models()

            if log:
                bar.set_postfix(average_result)
                bar.update()

                if epoch == epochs - 1:
                    self.report_results(results, "Joint Training")

        if plot:
            title = f"{model_type}"
            plot_path = f"{save_path}/plots/{now}/"
            plot_metrics(average_results, title=title, save_path=plot_path)

        if log:
            self.report_server_test()
        test_results = self.test_clients()
        average_result = sum_lod(test_results, coef)
        final_results = {}
        for cleint, test_result in zip(self.clients, test_results):
            final_results[f"Client{cleint.id}"] = test_result
        final_results["Average"] = average_result
        if log:
            self.report_test_results(final_results)

        return final_results

    def joint_train_w(
        self,
        epochs=config.model.iterations,
        log=True,
        plot=True,
        FL=True,
        model_type="GNN",
    ):
        if log:
            LOGGER.info(f"{model_type} starts!")
            bar = tqdm(total=epochs, position=0)

        num_nodes = sum([client.num_nodes() for client in self.clients])
        coef = [client.num_nodes() / num_nodes for client in self.clients]
        average_results = []
        for epoch in range(epochs):
            if FL:
                self.share_weights()
            self.reset_trainings()

            self.set_train_mode()
            results = self.train_clients(eval_=log)
            average_result = sum_lod(results, coef)
            average_result["Epoch"] = epoch + 1
            average_results.append(average_result)

            if FL:
                clients_grads = get_grads(self.clients, True)
                grads = sum_lod(clients_grads, coef)
                self.share_grads(grads)

            self.update_models()

            if FL:
                clients_weights = state_dict(self.clients)
                mean_weights = sum_lod(clients_weights, coef)
                self.load_state_dict(mean_weights)

            if log:
                bar.set_postfix(average_result)
                bar.update()

                if epoch == epochs - 1:
                    self.report_results(results, "Joint Training")

        if plot:
            title = f"{model_type}"
            plot_path = f"{save_path}/plots/{now}/"
            plot_metrics(average_results, title=title, save_path=plot_path)

        if log:
            self.report_server_test()
        test_results = self.test_clients()
        average_result = sum_lod(test_results, coef)
        final_results = {}
        for cleint, test_result in zip(self.clients, test_results):
            final_results[f"Client{cleint.id}"] = test_result
        final_results["Average"] = average_result
        if log:
            self.report_test_results(final_results)

        return final_results
