from ast import List
from copy import deepcopy
import numpy as np
import torch

from tqdm import tqdm
from src.classifier import Classifier

from src.utils import config
from src.client import Client
from src.utils.graph import Graph
from src.GNN_classifier import GNNClassifier
from src.MLP_classifier import MLPClassifier
from src.structure_predictor import StructurePredictor


class Server(Client):
    def __init__(
        self,
        graph: Graph,
        num_classes,
        logger=None,
    ):
        super().__init__(
            subgraph_=graph,
            num_classes=num_classes,
            id="Server",
            logger=logger,
        )

        self.sd_predictor = StructurePredictor(
            id=self.id,
            edge_index=self.subgraph.get_edges(),
            node_ids=self.subgraph.node_ids,
            y=self.subgraph.y,
            logger=logger,
        )

        self.clients: List[Client] = []
        self.num_clients = 0

        # self.subgraph.add_structural_features()

    def add_client(self, subgraph):
        client = Client(
            subgraph_=subgraph,
            id=self.num_clients,
            num_classes=self.num_classes,
            logger=self.LOGGER,
        )

        self.clients.append(client)
        self.num_clients += 1

    def train_local_gnns(self):
        for client in self.clients:
            client.train_local_gnn()
            self.LOGGER.info(
                f"Client{client.id} test accuracy: {client.test_local_gnn():0.4f}"
            )

    def train_local_mlps(self):
        for client in self.clients:
            client.train_local_mlp()
            self.LOGGER.info(
                f"Client{client.id} test accuracy: {client.test_local_mlp():0.4f}"
            )

    def add_weights(sum_weights, weights):
        if sum_weights is None:
            sum_weights = deepcopy(weights)
        else:
            for layer_name, layer_parameters in weights.items():
                for component_name, component_parameters in layer_parameters.items():
                    sum_weights[layer_name][component_name] += component_parameters
        return sum_weights

    def calc_mean_weights(sum_weights, count):
        for layer_name, layer_parameters in sum_weights.items():
            for component_name, component_parameters in layer_parameters.items():
                sum_weights[layer_name][component_name] = component_parameters / count

        return sum_weights

    def reset_sd_predictor_parameters(self):
        self.sd_predictor.reset_parameters()

    def get_sd_predictor_weights(self):
        return self.sd_predictor.state_dict()

    def set_sd_predictor_weights(self, weights):
        self.sd_predictor.load_state_dict(weights)

    def initialize_sd_predictor(self) -> None:
        sd_dims = [config.num_structural_features] + config.structure_layers_size

        self.create_local_sd_model(sd_dims)
        server_model = self.get_local_sd_model()

        client_models = []
        for client in self.clients:
            client.create_local_sd_model(sd_dims)
            client_models.append(client.get_local_sd_model())

        self.sd_predictor.initialize_structural_graph(
            client_models=client_models,
            server_model=server_model,
            sd_layer_size=sd_dims,
            num_classes=self.num_classes,
        )

    def get_sd_embeddings(self):
        return self.sd_predictor.get_embeddings()

    def fit_sd_predictor(self, epochs, bar=False, plot=False, predict=False) -> None:
        return self.sd_predictor.fit(epochs=epochs, bar=bar, plot=plot, predict=predict)

    def train_sd_predictor(
        self, epochs=config.epoch_classifier, bar=True, plot=True, predict=False
    ) -> None:
        self.initialize_sd_predictor()
        return self.fit_sd_predictor(epochs, bar=bar, plot=plot, predict=predict)

    def test_sd_predictor(self) -> None:
        test_acc = self.sd_predictor.test_cls()
        self.LOGGER.info(f"SD predictor test accuracy: {test_acc}")

    def train_SDSG(self, epochs=1):
        self.LOGGER.info("SDSG starts!")
        self.initialize_sd_predictor()
        plot_results = {}
        for client in self.clients:
            plot_results[f"Client{client.id}"] = []

        bar = tqdm(total=epochs)
        metrics = {}
        for epoch in range(epochs):
            res = self.sd_predictor.joint_train(self.clients, 1)
            metrics["Total Loss"] = res["Total Loss"]
            metrics["Structure Loss"] = res["Structure Loss"]

            for client in self.clients:
                result = res[f"client{client.id}"]
                result["Epoch"] = epoch
                plot_results[f"Client{client.id}"].append(result)
                metrics[f"Client{client.id}"] = result["Val Acc"]

            bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
            bar.set_postfix(metrics)
            bar.update()

        test_metrics = self.sd_predictor.test(self.clients)
        for client in self.clients:
            GNNClassifier.plot_results(
                plot_results[f"Client{client.id}"],
                client.id,
                type="SDSG",
            )

            result = res[f"client{client.id}"]
            self.LOGGER.info(f"SDSG results for client{client.id}:")
            self.LOGGER.info(f"{result}")

            test_acc = test_metrics[f"Client{client.id}"]["Test Acc"]

            self.LOGGER.info(f"Client{client.id} test accuracy: {test_acc}")

    def train_FLSW(self, epochs=config.epoch_classifier, model_type="GNN"):
        self.LOGGER.info("FLSW starts!")
        criterion = torch.nn.CrossEntropyLoss()
        plot_results = {}

        if model_type == "GNN":
            self.initialize_gnn()
            self.reset_gnn_parameters()
        elif model_type == "MLP":
            self.initialize_mlp()
            self.reset_mlp_parameters()

        plot_results[f"Client{self.id}"] = []

        for client in self.clients:
            if model_type == "GNN":
                client.initialize_gnn()
                client.reset_gnn_parameters()
            elif model_type == "MLP":
                client.initialize_mlp()
                client.reset_mlp_parameters()
            plot_results[f"Client{client.id}"] = []

        bar = tqdm(total=epochs)
        for epoch in range(epochs):
            metrics = {}
            if model_type == "GNN":
                weights = self.get_gnn_weights()
                (train_loss, train_acc, _, val_loss, val_acc, _) = GNNClassifier.train(
                    x=self.subgraph.x,
                    y=self.subgraph.y,
                    edge_index=self.subgraph.edge_index,
                    model=self.gnn_classifier.model,
                    criterion=criterion,
                    train_mask=self.subgraph.train_mask,
                    val_mask=self.subgraph.val_mask,
                )
            elif model_type == "MLP":
                weights = self.get_mlp_weights()
                (train_loss, train_acc, _, val_loss, val_acc, _) = MLPClassifier.train(
                    x=self.subgraph.x,
                    y=self.subgraph.y,
                    # client.subgraph.edge_index
                    model=self.mlp_classifier.model,
                    criterion=criterion,
                    train_mask=self.subgraph.train_mask,
                    val_mask=self.subgraph.val_mask,
                )

            result = {
                "Train Loss": round(train_loss.item(), 4),
                "Train Acc": round(train_acc, 4),
                "Val Loss": round(val_loss.item(), 4),
                "Val Acc": round(val_acc, 4),
                "Epoch": epoch,
            }

            metrics[f"client{self.id}"] = result["Val Acc"]
            plot_results[f"Client{self.id}"].append(result)
            if epoch == epochs - 1:
                self.LOGGER.info(f"FLSW results for client{self.id}:")
                self.LOGGER.info(f"{result}")

            sum_weights = None
            for client in self.clients:
                if model_type == "GNN":
                    client.set_gnn_weights(weights)
                    res = client.fit_gnn(config.epochs_local)
                    new_weights = client.get_gnn_weights()
                elif model_type == "MLP":
                    client.set_mlp_weights(weights)
                    res = client.fit_mlp(config.epochs_local)
                    new_weights = client.get_mlp_weights()

                sum_weights = Server.add_weights(sum_weights, new_weights)

                result = res[-1]
                result["Epoch"] = epoch
                metrics[f"client{client.id}"] = result["Val Acc"]
                plot_results[f"Client{client.id}"].append(result)
                if epoch == epochs - 1:
                    self.LOGGER.info(f"FLSW results for client{client.id}:")
                    self.LOGGER.info(f"{result}")

            mean_weights = Server.calc_mean_weights(sum_weights, self.num_clients)
            if model_type == "GNN":
                self.set_gnn_weights(mean_weights)
            elif model_type == "MLP":
                self.set_mlp_weights(mean_weights)
            bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
            bar.set_postfix(metrics)
            bar.update()

        if model_type == "GNN":
            self.LOGGER.info(f"Server test accuracy: {self.test_local_gnn():0.4f}")
        elif model_type == "MLP":
            self.LOGGER.info(f"Server test accuracy: {self.test_local_mlp():0.4f}")

        Classifier.plot_results(
            plot_results[f"Client{self.id}"],
            self.id,
            type="FLSW",
            model_type=model_type,
        )

        for client in self.clients:
            if model_type == "GNN":
                self.LOGGER.info(
                    f"Clinet{client.id} test accuracy: {client.test_local_gnn():0.4f}"
                )
            elif model_type == "MLP":
                self.LOGGER.info(
                    f"Clinet{client.id} test accuracy: {client.test_local_mlp():0.4f}"
                )

            Classifier.plot_results(
                plot_results[f"Client{client.id}"],
                client.id,
                type="FLSW",
                model_type=model_type,
            )

    def train_FLSG(self, epochs=1):
        # return self.model.fit2(self.graph, clients, epochs)
        self.LOGGER.info("FLSG starts!")
        plot_results = {}
        self.initialize_gnn()
        self.reset_gnn_parameters()
        plot_results[f"Client{self.id}"] = []

        GNN_model = self.gnn_classifier
        server_parameters = list(GNN_model.parameters())

        optimizer = torch.optim.Adam(server_parameters, lr=config.lr, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        for client in self.clients:
            client.initialize_gnn()
            client.reset_gnn_parameters()
            plot_results[f"Client{client.id}"] = []

        bar = tqdm(total=epochs)
        for epoch in range(epochs):
            optimizer.zero_grad()
            server_weights = GNN_model.state_dict()
            for client in self.clients:
                client.set_gnn_weights(server_weights)

            metrics = {}
            loss_list = torch.zeros(len(self.clients), dtype=torch.float32)

            # server
            (train_loss, train_acc, _, val_loss, val_acc, _) = MLPClassifier.train(
                x=self.subgraph.x,
                y=self.subgraph.y,
                # client.subgraph.edge_index
                model=self.mlp_classifier.model,
                criterion=criterion,
                train_mask=self.subgraph.train_mask,
                val_mask=self.subgraph.val_mask,
            )

            result = {
                "Train Loss": round(train_loss.item(), 4),
                "Train Acc": round(train_acc, 4),
                "Val Loss": round(val_loss.item(), 4),
                "Val Acc": round(val_acc, 4),
                "Epoch": epoch,
            }

            metrics[f"client{self.id}"] = result["Val Acc"]
            plot_results[f"Client{self.id}"].append(result)
            if epoch == epochs - 1:
                self.LOGGER.info(f"FLSG results for client{self.id}:")
                self.LOGGER.info(f"{result}")

            for ind, client in enumerate(self.clients):
                (
                    train_loss,
                    train_acc,
                    _,
                    val_loss,
                    val_acc,
                    _,
                ) = GNNClassifier.train(
                    x=client.subgraph.x,
                    y=client.subgraph.y,
                    edge_index=client.subgraph.edge_index,
                    model=client.gnn_classifier.model,
                    criterion=criterion,
                    train_mask=client.subgraph.train_mask,
                    val_mask=client.subgraph.val_mask,
                )
                loss_list[ind] = train_loss

                result = {
                    "Train Loss": round(train_loss.item(), 4),
                    "Train Acc": round(train_acc, 4),
                    "Val Loss": round(val_loss.item(), 4),
                    "Val Acc": round(val_acc, 4),
                    "Epoch": epoch,
                }

                # metrics[f"client{client.id}"] = result

                metrics[f"client{client.id}"] = result["Val Acc"]
                plot_results[f"Client{client.id}"].append(result)
                if epoch == epochs - 1:
                    self.LOGGER.info(f"FLSG results for client{client.id}:")
                    self.LOGGER.info(f"{result}")

            bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
            bar.set_postfix(metrics)
            bar.update()

            total_loss = loss_list.mean()
            total_loss.backward()
            grads = None
            for ind, client in enumerate(self.clients):
                client_parameters = list(client.gnn_classifier.parameters())
                client_grads = [
                    client_parameter.grad for client_parameter in client_parameters
                ]
                # ratio = client.subgraph.num_nodes / self.subgraph.num_nodes
                ratio = 1 / len(self.clients)
                if grads is None:
                    grads = [ratio * grad for grad in client_grads]
                else:
                    for i in range(len(client_grads)):
                        grads[i] += ratio * client_grads[i]

            for i in range(len(grads)):
                server_parameters[i].grad = grads[i]

            optimizer.step()
            # metrics["Total Loss"] = round(total_loss.item(), 4)

        self.LOGGER.info(f"Server test accuracy: {self.test_local_gnn():0.4f}")
        Classifier.plot_results(
            plot_results[f"Client{self.id}"],
            self.id,
            type="FLSG",
            model_type="GNN",
        )

        for client in self.clients:
            self.LOGGER.info(
                f"Clinet{client.id} test accuracy: {client.test_local_gnn():0.4f}"
            )

            Classifier.plot_results(
                plot_results[f"Client{client.id}"],
                client.id,
                type="FLSG",
                model_type="GNN",
            )

        return metrics

    def train_FLSG_MLP(self, epochs=1):
        # return self.model.fit2(self.graph, clients, epochs)
        self.LOGGER.info("FLSG MLP starts!")
        plot_results = {}
        self.initialize_mlp()
        self.reset_mlp_parameters()
        plot_results[f"Client{self.id}"] = []

        mlp_model = self.mlp_classifier
        server_parameters = list(mlp_model.parameters())

        optimizer = torch.optim.Adam(server_parameters, lr=config.lr, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        for client in self.clients:
            client.initialize_mlp()
            client.reset_mlp_parameters()
            plot_results[f"Client{client.id}"] = []

        bar = tqdm(total=epochs)
        for epoch in range(epochs):
            optimizer.zero_grad()
            server_weights = mlp_model.state_dict()
            for client in self.clients:
                client.set_mlp_weights(server_weights)

            metrics = {}
            loss_list = torch.zeros(len(self.clients), dtype=torch.float32)

            # server
            (train_loss, train_acc, _, val_loss, val_acc, _) = MLPClassifier.train(
                self.subgraph.x,
                self.subgraph.y,
                # client.subgraph.edge_index
                self.mlp_classifier.model,
                criterion,
                self.subgraph.train_mask,
                self.subgraph.val_mask,
            )

            result = {
                "Train Loss": round(train_loss.item(), 4),
                "Train Acc": round(train_acc, 4),
                "Val Loss": round(val_loss.item(), 4),
                "Val Acc": round(val_acc, 4),
                "Epoch": epoch,
            }

            metrics[f"client{self.id}"] = result["Val Acc"]
            plot_results[f"Client{self.id}"].append(result)
            if epoch == epochs - 1:
                self.LOGGER.info(f"FLSG results for client{self.id}:")
                self.LOGGER.info(f"{result}")

            for ind, client in enumerate(self.clients):
                (train_loss, train_acc, _, val_loss, val_acc, _) = MLPClassifier.train(
                    client.subgraph.x,
                    client.subgraph.y,
                    # client.subgraph.edge_index
                    client.mlp_classifier.model,
                    criterion,
                    client.subgraph.train_mask,
                    client.subgraph.val_mask,
                )

                result = {
                    "Train Loss": round(train_loss.item(), 4),
                    "Train Acc": round(train_acc, 4),
                    "Val Loss": round(val_loss.item(), 4),
                    "Val Acc": round(val_acc, 4),
                    "Epoch": epoch,
                }

                loss_list[ind] = train_loss

                # metrics[f"client{client.id}"] = result

                metrics[f"client{client.id}"] = result["Val Acc"]
                plot_results[f"Client{client.id}"].append(result)
                if epoch == epochs - 1:
                    self.LOGGER.info(f"FLSG results for client{client.id}:")
                    self.LOGGER.info(f"{result}")

            bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
            bar.set_postfix(metrics)
            bar.update()

            total_loss = loss_list.mean()
            total_loss.backward()
            grads = None
            for ind, client in enumerate(self.clients):
                client_parameters = list(client.mlp_classifier.parameters())
                client_grads = [
                    client_parameter.grad for client_parameter in client_parameters
                ]
                # ratio = client.subgraph.num_nodes / self.subgraph.num_nodes
                ratio = 1 / len(self.clients)
                if grads is None:
                    grads = [ratio * grad for grad in client_grads]
                else:
                    for i in range(len(client_grads)):
                        grads[i] += ratio * client_grads[i]

            for i in range(len(grads)):
                server_parameters[i].grad = grads[i]

            optimizer.step()
            # metrics["Total Loss"] = round(total_loss.item(), 4)

        self.LOGGER.info(f"Server test accuracy: {self.test_local_mlp():0.4f}")
        Classifier.plot_results(
            plot_results[f"Client{self.id}"],
            self.id,
            type="FLSG",
            model_type="MLP",
        )

        for client in self.clients:
            self.LOGGER.info(
                f"Clinet{client.id} test accuracy: {client.test_local_mlp():0.4f}"
            )

            Classifier.plot_results(
                plot_results[f"Client{client.id}"],
                client.id,
                type="FLSG",
                model_type="MLP",
            )

        return metrics

    def train_SDSW(self, epochs):
        self.LOGGER.info("SDSW starts!")
        self.initialize_sd_predictor()
        return self.sd_predictor.model.fit(
            self.sd_predictor.graph, self.clients, epochs
        )
