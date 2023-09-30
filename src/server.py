from ast import List

import torch
from tqdm import tqdm

from src.utils.utils import *
from src.client import Client
from src.utils.graph import Graph
from src.utils.config_parser import Config
from src.GNN_classifier import GNNClassifier
from src.MLP_classifier import MLPClassifier
from src.structure_predictor2 import StructurePredictor

config = Config()


class Server(Client):
    def __init__(
        self,
        graph: Graph,
        num_classes,
        classifier_type="GNN",
        save_path="./",
        logger=None,
    ):
        super().__init__(
            subgraph_=graph,
            num_classes=num_classes,
            id="Server",
            classifier_type=classifier_type,
            save_path=save_path,
            logger=logger,
        )
        self.clients: List[Client] = []
        self.num_clients = 0

        if self.classifier_type == "GNN":
            self.sd_predictor = StructurePredictor(
                id=self.id,
                edge_index=self.subgraph.get_edges(),
                node_ids=self.subgraph.node_ids,
                y=self.subgraph.y,
                masks=self.subgraph.get_masks(),
                save_path=self.save_path,
                logger=logger,
            )

            self.initialized = False

        # self.subgraph.add_structural_features()

    def add_client(self, subgraph):
        client = Client(
            subgraph_=subgraph,
            id=self.num_clients,
            num_classes=self.num_classes,
            classifier_type=self.classifier_type,
            save_path=self.save_path,
            logger=self.LOGGER,
        )

        self.clients.append(client)
        self.num_clients += 1

    def train_local_classifiers(self, epochs):
        plot_results = {}
        for client in self.clients:
            results = client.train_local_classifier(epochs, plot=False)
            plot_results[f"Client{client.id}"] = results

        average_results = []
        for i in range(epochs):
            average_result = {key: 0 for key in results[0].keys()}
            for client in self.clients:
                result = plot_results[f"Client{client.id}"][i]
                for key, val in result.items():
                    average_result[key] += val * client.num_nodes() / self.num_nodes()
                average_results.append(average_result)

        title = f"Average local {self.classifier_type}"
        plot_metrics(average_results, title=title, save_path=self.save_path)

        average_test_acc = 0
        for client in self.clients:
            test_acc = client.test_local_classifier()
            self.LOGGER.info(f"Client{client.id} test accuracy: {test_acc:0.4f}")
            average_test_acc += test_acc * client.num_nodes() / self.num_nodes()

        self.LOGGER.info(f"Average test accuracy: {average_test_acc:0.4f}")

    def reset_sd_predictor_parameters(self):
        self.sd_predictor.reset_parameters()

    def get_sd_predictor_weights(self):
        return self.sd_predictor.state_dict()

    def set_sd_predictor_weights(self, weights):
        self.sd_predictor.load_state_dict(weights)

    # def initialize_sd_predictor(self) -> None:
    #     sd_dims = [
    #         config.structure_model.num_structural_features
    #     ] + config.structure_model.structure_layers_size

    #     self.create_local_sd_model(sd_dims)
    #     server_model = self.get_local_sd_model()

    #     client_models = []
    #     for client in self.clients:
    #         client.create_local_sd_model(sd_dims)
    #         client_models.append(client.get_local_sd_model())

    #     self.sd_predictor.initialize_structural_graph(
    #         client_models=client_models,
    #         server_model=server_model,
    #         sd_layer_size=sd_dims,
    #         num_classes=self.num_classes,
    #     )

    def initialize_sd_predictor(
        self,
        propagate_type=config.model.propagate_type,
    ) -> None:
        if propagate_type == "MP":
            if config.structure_model.structure_type == "random":
                additional_layer_dims = config.structure_model.num_structural_features
            else:
                additional_layer_dims = (
                    config.structure_model.MP_structure_layers_sizes[-1]
                )
        elif propagate_type == "GNN":
            additional_layer_dims = config.structure_model.GNN_structure_layers_sizes[
                -1
            ]

        self.initialize(
            propagate_type=propagate_type,
            additional_layer_dims=additional_layer_dims,
        )
        client: Client
        for client in self.clients:
            client.initialize(
                propagate_type=propagate_type,
                additional_layer_dims=additional_layer_dims,
            )

        self.sd_predictor.initialize_structural_graph(
            client_models=self.clients,
            server_model=self,
            propagate_type=propagate_type,
        )

        self.initialized = True

    def train(self, mode: bool = True):
        self.sd_predictor.train(mode)
        super().train(mode)

    def eval(self):
        self.sd_predictor.eval()
        super().eval()

    def calc_total_grads(self, clients):
        grads = None
        for client in self.clients:
            client_grads = client.classifier.get_model_grads()
            ratio = client.num_nodes() / self.num_nodes()
            if grads is None:
                grads = [ratio * grad for grad in client_grads]
            else:
                for i in range(len(client_grads)):
                    grads[i] += ratio * client_grads[i]

        server_parameters = list(self.classifier.parameters())
        for i in range(len(grads)):
            server_parameters[i].grad = grads[i]

    def get_structure_embeddings(self):
        return self.sd_predictor.get_structure_embeddings()

    def fit_sd_predictor(self, epochs, bar=False, plot=False, predict=False) -> None:
        return self.sd_predictor.fit(epochs=epochs, bar=bar, plot=plot, predict=predict)

    def train_sd_predictor(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        bar=True,
        plot=True,
        predict=False,
    ) -> None:
        if self.initialized:
            self.reset_sd_predictor_parameters()
        else:
            self.initialize_sd_predictor(propagate_type=propagate_type)
        return self.fit_sd_predictor(epochs, bar=bar, plot=plot, predict=predict)

    def test_sd_predictor(self) -> None:
        test_acc = self.sd_predictor.test_cls()
        self.LOGGER.info(f"SD predictor test accuracy: {test_acc}")

    def train_FLWA(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        log=True,
        plot=True,
    ):
        if log:
            self.LOGGER.info("FLWA starts!")
        final_result = {}
        criterion = torch.nn.CrossEntropyLoss()

        self.initialize(propagate_type=propagate_type)
        self.reset_parameters()

        client: Client
        for client in self.clients:
            client.initialize(propagate_type=propagate_type)
            client.reset_parameters()

        server_results = []
        average_results = []
        if log:
            bar = tqdm(total=epochs)
        for epoch in range(epochs):
            metrics = {}
            weights = self.state_dict()
            self.classifier.eval()
            if self.classifier_type == "GNN":
                (train_loss, train_acc, _, val_loss, val_acc, _) = GNNClassifier.step(
                    x=self.subgraph.x,
                    y=self.subgraph.y,
                    edge_index=self.subgraph.edge_index,
                    model=self.classifier.model,
                    criterion=criterion,
                    train_mask=self.subgraph.train_mask,
                    val_mask=self.subgraph.val_mask,
                )
            elif self.classifier_type == "MLP":
                (train_loss, train_acc, _, val_loss, val_acc, _) = MLPClassifier.step(
                    x=self.subgraph.x,
                    y=self.subgraph.y,
                    # client.subgraph.edge_index
                    model=self.classifier.model,
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
            server_results.append(result)

            if log:
                metrics[f"server train acc"] = result["Train Acc"]
                metrics[f"server val acc"] = result["Val Acc"]
                if epoch == epochs - 1:
                    self.LOGGER.info(f"FLWA results for client{self.id}:")
                    self.LOGGER.info(f"{result}")

            for client in self.clients:
                client.load_state_dict(weights)

            sum_weights = None
            average_result = {}
            for client in self.clients:
                res = client.fit(config.model.epochs_local)

                new_weights = client.state_dict()
                sum_weights = add_weights(sum_weights, new_weights)

                result = res[-1]

                ratio = client.num_nodes() / self.num_nodes()
                for key, val in result.items():
                    if key not in average_result.keys():
                        average_result[key] = ratio * val
                    else:
                        average_result[key] += ratio * val

                if log:
                    if epoch == epochs - 1:
                        self.LOGGER.info(f"FLWA results for client{client.id}:")
                        self.LOGGER.info(f"{result}")

            mean_weights = calc_mean_weights(sum_weights, self.num_clients)
            self.load_state_dict(mean_weights)

            average_result["Epoch"] = epoch + 1
            average_results.append(average_result)

            if log:
                metrics[f"average train acc"] = average_result["Train Acc"]
                metrics[f"average val acc"] = average_result["Val Acc"]
                bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
                bar.set_postfix(metrics)
                bar.update()

        test_acc = self.test_local_classifier()
        final_result["Server"] = test_acc

        if log:
            self.LOGGER.info(f"Server test accuracy: {test_acc:0.4f}")

        average_test_acc = 0
        for client in self.clients:
            test_acc = client.test_local_classifier()
            final_result[f"Clinet{client.id}"] = test_acc
            if log:
                self.LOGGER.info(f"Clinet{client.id} test accuracy: {test_acc:0.4f}")
            average_test_acc += test_acc * client.num_nodes() / self.num_nodes()

        final_result["Average"] = average_test_acc
        if log:
            self.LOGGER.info(f"Average test accuracy: {average_test_acc:0.4f}")

        if plot:
            title = f"Server FLWA {self.classifier_type}"
            plot_metrics(server_results, title=title, save_path=self.save_path)

            title = f"Average FLWA {self.classifier_type}"
            plot_metrics(average_results, title=title, save_path=self.save_path)

        return final_result

    def train_FLGA(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        log=True,
        plot=True,
    ):
        # return self.model.fit2(self.graph, clients, epochs)
        self.LOGGER.info("FLGA starts!")
        final_result = {}
        self.initialize(propagate_type=propagate_type)
        self.reset_parameters()

        client: Client
        for client in self.clients:
            client.initialize(propagate_type=propagate_type)
            client.reset_parameters()

        classifier = self.classifier
        server_parameters = list(classifier.parameters())

        optimizer = torch.optim.Adam(
            server_parameters,
            lr=config.model.lr,
            weight_decay=config.model.weight_decay,
        )
        criterion = torch.nn.CrossEntropyLoss()

        server_results = []
        average_results = []

        if log:
            bar = tqdm(total=epochs)
        for epoch in range(epochs):
            # server
            self.classifier.eval()
            if self.classifier_type == "GNN":
                (train_loss, train_acc, _, val_loss, val_acc, _) = GNNClassifier.step(
                    x=self.subgraph.x,
                    y=self.subgraph.y,
                    edge_index=client.subgraph.edge_index,
                    model=self.classifier.model,
                    criterion=criterion,
                    train_mask=self.subgraph.train_mask,
                    val_mask=self.subgraph.val_mask,
                )
            elif self.classifier_type == "MLP":
                (train_loss, train_acc, _, val_loss, val_acc, _) = MLPClassifier.step(
                    x=self.subgraph.x,
                    y=self.subgraph.y,
                    # edge_index=client.subgraph.edge_index,
                    model=self.classifier.model,
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

            server_results.append(result)
            metrics = {}
            if log:
                metrics[f"server train acc"] = result["Train Acc"]
                metrics[f"server val acc"] = result["Val Acc"]
                if epoch == epochs - 1:
                    self.LOGGER.info(f"FLGA results for client{self.id}:")
                    self.LOGGER.info(f"{result}")

            optimizer.zero_grad(set_to_none=True)
            server_weights = classifier.state_dict()

            for client in self.clients:
                client.eval()
                client.classifier.zero_grad()
                client.load_state_dict(server_weights)

            loss_list = torch.zeros(len(self.clients), dtype=torch.float32)
            self.classifier.zero_grad(set_to_none=True)

            average_result = {}
            for ind, client in enumerate(self.clients):
                client.train()
                if self.classifier_type == "GNN":
                    (
                        train_loss,
                        train_acc,
                        _,
                        val_loss,
                        val_acc,
                        _,
                    ) = GNNClassifier.step(
                        x=client.subgraph.x,
                        y=client.subgraph.y,
                        edge_index=client.subgraph.edge_index,
                        model=client.classifier.model,
                        criterion=criterion,
                        train_mask=client.subgraph.train_mask,
                        val_mask=client.subgraph.val_mask,
                    )
                elif self.classifier_type == "MLP":
                    (
                        train_loss,
                        train_acc,
                        _,
                        val_loss,
                        val_acc,
                        _,
                    ) = MLPClassifier.step(
                        x=client.subgraph.x,
                        y=client.subgraph.y,
                        # edge_index=client.subgraph.edge_index,
                        model=client.classifier.model,
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
                }

                ratio = client.num_nodes() / self.num_nodes()
                for key, val in result.items():
                    if key not in average_result.keys():
                        average_result[key] = ratio * val
                    else:
                        average_result[key] += ratio * val

                if log:
                    if epoch == epochs - 1:
                        self.LOGGER.info(f"FLGA results for client{client.id}:")
                        self.LOGGER.info(f"{result}")

            average_result["Epoch"] = epoch + 1
            average_results.append(average_result)

            if log:
                metrics[f"average train acc"] = average_result["Train Acc"]
                metrics[f"average val acc"] = average_result["Val Acc"]
                bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
                bar.set_postfix(metrics)
                bar.update()

            total_loss = loss_list.mean()
            total_loss.backward()
            self.calc_total_grads(self.clients)
            optimizer.step()
            # metrics["Total Loss"] = round(total_loss.item(), 4)

        test_acc = self.test_local_classifier()
        final_result["Server"] = test_acc
        if log:
            self.LOGGER.info(f"Server test accuracy: {test_acc:0.4f}")

        average_test_acc = 0
        for client in self.clients:
            test_acc = client.test_local_classifier()
            final_result[f"Clinet{client.id}"] = test_acc
            if log:
                self.LOGGER.info(f"Clinet{client.id} test accuracy: {test_acc:0.4f}")

            average_test_acc += test_acc * client.num_nodes() / self.num_nodes()

        final_result["Average"] = average_test_acc
        if log:
            self.LOGGER.info(f"Average test accuracy: {average_test_acc:0.4f}")

        if plot:
            title = f"Server FLGA {self.classifier_type}"
            plot_metrics(server_results, title=title, save_path=self.save_path)

            title = f"Average FLGA {self.classifier_type}"
            plot_metrics(average_results, title=title, save_path=self.save_path)

        return final_result

    def train_SD_Server(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        log=True,
        plot=True,
    ):
        if log:
            self.LOGGER.info("SD_Server starts!")
        if self.initialized:
            self.reset_sd_predictor_parameters()
        else:
            self.initialize_sd_predictor(propagate_type=propagate_type)
        return self.sd_predictor.train_SD_Server(
            epochs=epochs,
            log=log,
            plot=plot,
        )

    def train_SDWA(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        log=True,
        plot=True,
    ):
        self.LOGGER.info("SDWA starts!")
        if self.initialized:
            self.reset_sd_predictor_parameters()
        else:
            self.initialize_sd_predictor(propagate_type=propagate_type)
        return self.sd_predictor.train_SDWA(
            self.clients,
            epochs=epochs,
            log=log,
            plot=plot,
        )

    def train_SDGA(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        log=True,
        plot=True,
    ):
        self.LOGGER.info("SDGA starts!")
        if self.initialized:
            self.reset_sd_predictor_parameters()
        else:
            self.initialize_sd_predictor(propagate_type=propagate_type)
        return self.sd_predictor.train_SDGA(
            self.clients,
            epochs=epochs,
            log=log,
            plot=plot,
        )

    def train_local_sd(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        log=True,
        plot=True,
    ):
        self.LOGGER.info("Local SD starts!")
        if self.initialized:
            self.reset_sd_predictor_parameters()
        else:
            self.initialize_sd_predictor(propagate_type=propagate_type)
        self.sd_predictor.train_local_sd(
            self.clients,
            epochs=epochs,
            log=log,
            plot=plot,
        )

    def train_locsages(self, log=True, plot=True):
        client: Client
        for client in self.clients:
            self.LOGGER.info(f"locsage for client{client.id}")
            client.train_locsage(log=log, plot=plot)
            self.LOGGER.info(
                f"Client{client.id} test accuracy: {client.test_local_classifier()}"
            )

    def train_fedgen(self):
        client: Client
        other_client: Client
        for client in self.clients:
            inter_client_features_creators = []
            for other_client in self.clients:
                if other_client.id != client.id:
                    inter_client_features_creators.append(
                        other_client.create_inter_features
                    )

            client.train_neighgen(inter_client_features_creators)

    def train_fedSage_plus(self, epochs=config.model.epoch_classifier):
        self.LOGGER.info("FedSage+ starts!")
        criterion = torch.nn.CrossEntropyLoss()
        self.initialize()
        self.reset_parameters()

        client: Client
        other_client: Client
        for client in self.clients:
            inter_client_features_creators = []
            for other_client in self.clients:
                if other_client.id != client.id:
                    inter_client_features_creators.append(
                        other_client.create_inter_features
                    )

            client.initialize_locsage(inter_client_features_creators)
            client.reset_parameters()

        server_results = []
        average_results = []

        bar = tqdm(total=epochs)
        for epoch in range(epochs):
            weights = self.state_dict()
            metrics = {}
            self.classifier.eval()
            if self.classifier_type == "GNN":
                (train_loss, train_acc, _, val_loss, val_acc, _) = GNNClassifier.step(
                    x=self.subgraph.x,
                    y=self.subgraph.y,
                    edge_index=self.subgraph.edge_index,
                    model=self.classifier.model,
                    criterion=criterion,
                    train_mask=self.subgraph.train_mask,
                    val_mask=self.subgraph.val_mask,
                )
            elif self.classifier_type == "MLP":
                (train_loss, train_acc, _, val_loss, val_acc, _) = MLPClassifier.step(
                    x=self.subgraph.x,
                    y=self.subgraph.y,
                    # client.subgraph.edge_index
                    model=self.classifier.model,
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

            metrics[f"server train acc"] = result["Train Acc"]
            metrics[f"server val acc"] = result["Val Acc"]
            server_results.append(result)
            if epoch == epochs - 1:
                self.LOGGER.info(f"fedsage+ results for client{self.id}:")
                self.LOGGER.info(f"{result}")

            sum_weights = None
            average_result = {}
            for client in self.clients:
                client.load_state_dict(weights)
                res = client.fit(config.model.epochs_local)
                new_weights = client.state_dict()
                sum_weights = add_weights(sum_weights, new_weights)

                result = res[-1]

                ratio = client.num_nodes() / self.num_nodes()
                for key, val in result.items():
                    if key not in average_result.keys():
                        average_result[key] = ratio * val
                    else:
                        average_result[key] += ratio * val

                if epoch == epochs - 1:
                    self.LOGGER.info(f"fedsage+ results for client{client.id}:")
                    self.LOGGER.info(f"{result}")

            mean_weights = calc_mean_weights(sum_weights, self.num_clients)
            self.load_state_dict(mean_weights)

            average_result["Epoch"] = epoch + 1
            average_results.append(average_result)

            metrics[f"average train acc"] = average_result["Train Acc"]
            metrics[f"average val acc"] = average_result["Val Acc"]

            bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
            bar.set_postfix(metrics)
            bar.update()

        self.LOGGER.info(f"Server test accuracy: {self.test_local_classifier():0.4f}")

        average_test_acc = 0
        for client in self.clients:
            test_acc = client.test_local_classifier()
            self.LOGGER.info(f"Clinet{client.id} test accuracy: {test_acc:0.4f}")

            average_test_acc += test_acc * client.num_nodes() / self.num_nodes()
        self.LOGGER.info(f"Average test accuracy: {average_test_acc:0.4f}")

        title = f"Server fedsage+ {self.classifier_type}"
        plot_metrics(server_results, title=title, save_path=self.save_path)

        title = f"Average fedsage+ {self.classifier_type}"
        plot_metrics(average_results, title=title, save_path=self.save_path)
