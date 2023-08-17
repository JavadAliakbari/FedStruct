from itertools import product
import logging

import torch
import numpy as np
from tqdm import tqdm
from torch.nn.functional import normalize
from torch_geometric.nn import MessagePassing

from src.utils.utils import *
from src.utils.graph import Graph
from src.utils.config_parser import Config
from src.models.structure_model3 import JointModel
from src.models.GNN_models import GNN, MLP, calc_accuracy

config = Config()


class StructurePredictor:
    def __init__(
        self,
        id,
        edge_index,
        node_ids,
        y=None,
        masks=None,
        save_path="./",
        logger=None,
    ):
        self.LOGGER = logger or logging

        self.id = id
        self.edge_index = edge_index
        self.node_ids = node_ids
        self.y = y
        self.masks = masks
        self.save_path = save_path

        self.message_passing = MessagePassing(aggr="mean")

        num_classes = max(y).item() + 1
        self.cls = MLP(
            [config.structure_model.structure_layers_sizes[-1], 32, num_classes],
            dropout=0.1,
            batch_normalization=True,
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def prepare_data(self):
        self.graph = Graph(
            edge_index=self.edge_index,
            node_ids=self.node_ids,
            y=self.y,
        )
        if self.masks is None:
            self.graph.add_masks()
        else:
            self.graph.set_masks(self.masks)

        self.graph.add_structural_features(
            structure_type=config.structure_model.structure_type,
            num_structural_features=config.structure_model.num_structural_features,
        )

        if self.graph.y is not None:
            self.graph.find_class_neighbors()

    def set_model(self, client_models, server_model, input_size):
        self.server = server_model

        self.structure_model = GNN(
            layer_sizes=input_size,
            last_layer="linear",
            layer_type="sage",
            dropout=config.model.dropout,
            batch_normalization=True,
            multiple_features=config.structure_model.structure_type == "mp",
            feature_dims=config.structure_model.num_mp_vectors,
        )

        self.model = JointModel(
            server=server_model,
            clients=client_models,
            logger=self.LOGGER,
        )

    def initialize_structural_graph(self, client_models, server_model, sd_layer_size):
        self.prepare_data()
        self.set_model(
            client_models,
            server_model,
            sd_layer_size,
        )

    def reset_parameters(self):
        self.model.reset_parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, weights):
        self.model.load_state_dict(weights)

    def cosine_similarity(h1, h2):
        return torch.dot(h1, h2) / (
            (torch.norm(h1) + 0.000001) * (torch.norm(h2) + 0.000001)
        )

    def calc_homophily_loss(self, embeddings):
        node_ids = self.graph.node_ids.numpy()
        edge_index = self.graph.edge_index

        normalized_embeddings = normalize(embeddings, dim=1)

        neighbors_embeddings = self.message_passing.propagate(
            edge_index, x=normalized_embeddings
        )

        negative_edge_index = []
        negative_sample_size = 10
        # negative_samples_list = []
        for node_id in node_ids:
            other_nodes = self.graph.negative_samples[node_id]

            negative_samples = np.random.choice(other_nodes, negative_sample_size)
            # negative_samples_list.append(negative_samples)
            negative_edge_index += list(product(negative_samples, [node_id]))

        negative_edge_index = torch.tensor(np.array(negative_edge_index).T)
        negative_embeddings = self.message_passing.propagate(
            negative_edge_index, x=normalized_embeddings
        )

        diff = negative_embeddings - neighbors_embeddings

        # loss1 = torch.einsum("ij,ij->i", neighbors_embeddings, normalized_embeddings)
        # loss2 = torch.einsum("ij,ij->i", negative_embeddings, normalized_embeddings)
        loss = (1 + torch.einsum("ij,ij->i", diff, normalized_embeddings)) / 2

        return loss

    def calc_homophily_loss2(self, embeddings):
        node_ids = self.graph.node_ids.numpy()

        loss_list = torch.zeros(len(node_ids), dtype=torch.float32)
        negative_sample_size = 5
        negative_mass_probability = self.graph.degree / sum(self.graph.degree)

        for node_idx, node_id in enumerate(node_ids):
            node_embedding = embeddings[node_id]
            neighbor_embeddings = embeddings[
                np.array(self.graph.node_neighbors[node_id])
            ]
            neighbor_loss = torch.log(
                torch.sigmoid(
                    torch.einsum(
                        "j,ij->i",
                        node_embedding,
                        neighbor_embeddings,
                    )
                ),
            ).sum()

            node_negative_sample_size = (
                self.graph.degree[node_id] * negative_sample_size
            )
            negative_samples = np.random.choice(
                node_ids,
                # negative_sample_size,
                node_negative_sample_size,
                p=negative_mass_probability,
                replace=False,
            )

            negative_embeddings = embeddings[negative_samples]
            negative_loss = torch.log(
                torch.sigmoid(
                    torch.einsum(
                        "j,ij->i",
                        node_embedding,
                        negative_embeddings,
                    )
                ),
            ).sum()

            loss_list[node_idx] = neighbor_loss - negative_loss

        return loss_list

    def calc_hetero_loss(self, embeddings):
        node_ids = self.graph.node_ids.numpy()
        y = self.graph.y.numpy()

        normalized_embeddings = normalize(embeddings, dim=1)

        neighbor_edge_index = []
        negative_edge_index = []
        sample_size = 15
        for node_id, class_ in zip(node_ids, y):
            class_nodes = self.graph.class_groups[class_]
            nieghbor_class_nodes = class_nodes[class_nodes != node_id]
            other_nodes = self.graph.negative_class_groups[class_]

            if class_ != 0:
                neighbor_samples = np.random.choice(nieghbor_class_nodes, sample_size)
                neighbor_edge_index += list(product(neighbor_samples, [node_id]))

            negative_samples = np.random.choice(other_nodes, 3 * sample_size)
            negative_edge_index += list(product(negative_samples, [node_id]))

        neighbor_edge_index = torch.tensor(np.array(neighbor_edge_index).T)
        neighbor_embeddings = self.message_passing.propagate(
            neighbor_edge_index, x=normalized_embeddings
        )

        negative_edge_index = torch.tensor(np.array(negative_edge_index).T)
        negative_embeddings = self.message_passing.propagate(
            negative_edge_index, x=normalized_embeddings
        )

        diff = negative_embeddings - neighbor_embeddings

        # loss1 = torch.einsum("ij,ij->i", neighbor_embeddings, normalized_embeddings)
        # loss2 = torch.einsum("ij,ij->i", negative_embeddings, normalized_embeddings)
        loss = (1 + torch.einsum("ij,ij->i", diff, normalized_embeddings)) / 2

        return loss

    def calc_loss(self, structure_embeddings):
        if config.structure_model.loss == "heterophily":
            loss = self.calc_hetero_loss(structure_embeddings)
        elif config.structure_model.loss == "homophily":
            loss = self.calc_homophily_loss(structure_embeddings)
        elif config.structure_model.loss == "both":
            loss1 = self.calc_hetero_loss(structure_embeddings)
            loss2 = self.calc_homophily_loss(structure_embeddings)
            loss = loss1 + loss2
        else:
            loss = torch.zeros(self.server.num_nodes(), dtype=torch.float32)

        return loss

    def get_structure_embeddings(self, train=True):
        if train:
            self.structure_model.train()
        else:
            self.structure_model.eval()
        x = self.graph.structural_features
        edge_index = self.graph.edge_index
        S = self.structure_model(x, edge_index)

        return S

    def fit(
        self,
        epochs=config.model.gen_epochs,
        plot=False,
        bar=False,
        predict=False,
    ):
        optimizer = torch.optim.Adam(
            self.structure_model.parameters(),
            lr=config.model.lr,
            weight_decay=config.model.weight_decay,
        )
        if bar:
            bar = tqdm(total=epochs, position=0)
        res = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            S = self.get_structure_embeddings(train=True)
            loss = self.calc_loss(S)

            train_loss = loss[self.graph.train_mask].mean()
            val_loss = loss[self.graph.val_mask].mean()

            train_loss.backward()
            optimizer.step()

            if predict:
                with torch.no_grad():
                    x = self.get_structure_embeddings(train=False)
                y = self.y
                masks = (
                    self.graph.train_mask,
                    self.graph.val_mask,
                    self.graph.test_mask,
                )
                self.cls.reset_parameters()
                cls_val_acc, cls_val_loss = self.cls.fit(x, y, masks, 100)
            else:
                cls_val_acc = 0
                cls_val_loss = 0

            metrics = {
                "Train Loss": round(train_loss.item(), 4),
                "Val Loss": round(val_loss.item(), 4),
                "CLS Val Acc": round(cls_val_acc, 4),
                "CLS Val Loss": round(cls_val_loss.item(), 4),
            }

            if bar:
                bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
                bar.set_postfix(metrics)
                bar.update(1)

            metrics["Epoch"] = epoch + 1
            res.append(metrics)

        # Test
        if plot:
            title = "cosine similarity embedding metrics"
            plot_metrics(res, title=title, save_path=self.save_path)

        return res

    @torch.no_grad()
    def test_cls(self):
        x = self.get_structure_embeddings(train=False)
        y = self.y
        test_acc = self.cls.test(x[self.graph.test_mask], y[self.graph.test_mask])
        return test_acc

    @torch.no_grad()
    def test(self, clients):
        self.model.eval()
        metrics = {}

        y = self.server.subgraph.y
        test_mask = self.server.subgraph.test_mask
        y_pred = self.model.step(self.server, train=False)
        test_acc = calc_accuracy(
            y_pred[test_mask].argmax(dim=1),
            y[test_mask],
        )

        result = {
            "Test Acc": round(test_acc, 4),
        }

        metrics[f"Client{self.server.id}"] = result

        if len(clients) > 0:
            out = self.model()

        for client in clients:
            y = client.subgraph.y
            test_mask = client.subgraph.test_mask

            y_pred = out[f"client{client.id}"]

            test_acc = calc_accuracy(
                y_pred[test_mask].argmax(dim=1),
                y[test_mask],
            )

            result = {
                "Test Acc": round(test_acc, 4),
            }

            metrics[f"Client{client.id}"] = result

        return metrics

    def cacl_metrics(self, client, y_pred, structure_loss):
        node_ids = client.get_nodes()
        y = client.subgraph.y
        train_mask = client.subgraph.train_mask
        val_mask = client.subgraph.val_mask

        client_structure_loss = structure_loss[node_ids]
        # y_pred = out[f"client{client.id}"]

        cls_train_loss = self.criterion(y_pred[train_mask], y[train_mask])
        str_train_loss = client_structure_loss[train_mask].mean()
        # loss_list[ind] = train_loss

        train_acc = calc_accuracy(
            y_pred[train_mask].argmax(dim=1),
            y[train_mask],
        )

        # Validation
        self.model.eval()
        with torch.no_grad():
            cls_val_loss = self.criterion(y_pred[val_mask], y[val_mask])
            str_val_loss = client_structure_loss[val_mask].mean()
            # val_loss = cls_val_loss + str_val_loss

            val_acc = calc_accuracy(y_pred[val_mask].argmax(dim=1), y[val_mask])

        return (
            cls_train_loss,
            str_train_loss,
            train_acc,
            cls_val_loss,
            str_val_loss,
            val_acc,
        )

    def step(self, train=True):
        S = self.get_structure_embeddings(train=train)
        h = self.server.get_feature_embeddings()
        x = torch.hstack((h, S))
        y_pred = self.server.predict_labels(x)

        structure_loss = self.calc_loss(S)

        return self.cacl_metrics(self.server, y_pred, structure_loss)

    def calc_server_weights(self, clients):
        sum_weights = None
        for client in clients:
            client_weight = client.state_dict()
            sum_weights = add_weights(sum_weights, client_weight)

        mean_weights = calc_mean_weights(sum_weights, len(clients))

        self.server.load_state_dict(mean_weights)

    def calc_total_grads(self, clients):
        grads = None
        for client in clients:
            client_parameters = client.parameters()
            client_grads = [
                client_parameter.grad for client_parameter in client_parameters
            ]
            ratio = client.num_nodes() / self.server.num_nodes()
            if grads is None:
                grads = [ratio * grad for grad in client_grads]
            else:
                for i in range(len(client_grads)):
                    grads[i] += ratio * client_grads[i]

        server_parameters = list(self.server.parameters())
        for i in range(len(grads)):
            server_parameters[i].grad = grads[i]

    def create_SW_optimizers(self):
        optimizers = {}
        for client in self.model:
            parameters = client.parameters()
            # parameters += list(self.models[f"structure_model"].parameters())
            optimizer = torch.optim.Adam(
                parameters, lr=config.model.lr, weight_decay=config.model.weight_decay
            )
            optimizers[f"client{client.id}"] = optimizer

        parameters = self.structure_model.parameters()
        optimizer = torch.optim.Adam(
            parameters, lr=config.model.lr, weight_decay=config.model.weight_decay
        )
        optimizers[f"structure_model"] = optimizer

        return optimizers

    def create_SG_optimizer(self):
        parameters = list(self.server.parameters()) + list(
            self.structure_model.parameters()
        )

        optimizer = torch.optim.Adam(
            parameters, lr=config.model.lr, weight_decay=config.model.weight_decay
        )

        return optimizer

    def reset_optimizers(optimizers) -> None:
        for optimizer in optimizers.values():
            optimizer.zero_grad()

    def train_SD_Server(self, epochs):
        optimizer = self.create_SG_optimizer()
        metrics = {}
        plot_results = {}
        plot_results[f"Client{self.server.id}"] = []

        bar = tqdm(total=epochs, position=0)
        for epoch in range(epochs):
            optimizer.zero_grad()

            (
                cls_train_loss,
                str_train_loss,
                train_acc,
                cls_val_loss,
                str_val_loss,
                val_acc,
            ) = self.step(train=True)

            result = {
                "Cls Train Loss": round(cls_train_loss.item(), 4),
                "Train Acc": round(train_acc, 4),
                "Cls Val Loss": round(cls_val_loss.item(), 4),
                "Val Acc": round(val_acc, 4),
                "Str Train Loss": round(str_train_loss.item(), 4),
                "Str Val Loss": round(str_val_loss.item(), 4),
                "Epoch": epoch,
            }

            plot_results[f"Client{self.server.id}"].append(result)
            metrics["Val ACC"] = result["Val Acc"]
            metrics["Cls Val Loss"] = result["Cls Val Loss"]
            metrics["Str Val Loss"] = result["Str Val Loss"]
            if epoch == epochs - 1:
                self.LOGGER.info(f"SD_Server results for client{self.server.id}:")
                self.LOGGER.info(f"{result}")

            train_loss = cls_train_loss + str_train_loss

            train_loss.backward()
            optimizer.step()

            bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
            bar.set_postfix(metrics)
            bar.update()

        title = f"Central SDSFL GNN"
        plot_metrics(
            plot_results[f"Client{self.server.id}"],
            title=title,
            save_path=self.save_path,
        )

        test_metrics = self.test([])
        test_acc = test_metrics[f"Client{self.server.id}"]["Test Acc"]
        self.LOGGER.info(f"Server test accuracy: {test_acc:0.4f}")
        if config.structure_model.structure_type == "mp":
            print(self.structure_model.mp_layer.state_dict()["weight"])

    def train_SDWA(self, clients, epochs=1):
        optimizers = self.create_SW_optimizers()
        metrics = {}
        server_results = []
        average_results = []

        bar = tqdm(total=epochs, position=0)
        for epoch in range(epochs):
            StructurePredictor.reset_optimizers(optimizers)

            (
                cls_train_loss,
                str_train_loss,
                train_acc,
                cls_val_loss,
                str_val_loss,
                val_acc,
            ) = self.step(train=False)

            result = {
                "Cls Train Loss": round(cls_train_loss.item(), 4),
                "Train Acc": round(train_acc, 4),
                "Cls Val Loss": round(cls_val_loss.item(), 4),
                "Val Acc": round(val_acc, 4),
                "Str Train Loss": round(str_train_loss.item(), 4),
                "Str Val Loss": round(str_val_loss.item(), 4),
                "Epoch": epoch,
            }

            server_results.append(result)

            metrics[f"server train acc"] = result["Train Acc"]
            metrics[f"server val acc"] = result["Val Acc"]
            if epoch == epochs - 1:
                self.LOGGER.info(f"SDWA results for {self.server.id}:")
                self.LOGGER.info(f"{result}")

            self.server.zero_grad()
            server_weights = self.server.state_dict()
            for client in clients:
                client.load_state_dict(server_weights)

            out = self.model()
            structure_loss = self.calc_loss(out["structure_model"])

            loss_list = torch.zeros(len(clients), dtype=torch.float32)
            average_result = {}
            for ind, client in enumerate(clients):
                y_pred = out[f"client{client.id}"]
                (
                    cls_train_loss,
                    str_train_loss,
                    train_acc,
                    cls_val_loss,
                    str_val_loss,
                    val_acc,
                ) = self.cacl_metrics(client, y_pred, structure_loss)

                train_loss = cls_train_loss + str_train_loss
                loss_list[ind] = train_loss

                result = {
                    "Cls Train Loss": round(cls_train_loss.item(), 4),
                    "Train Acc": round(train_acc, 4),
                    "Cls Val Loss": round(cls_val_loss.item(), 4),
                    "Val Acc": round(val_acc, 4),
                    "Str Train Loss": round(str_train_loss.item(), 4),
                    "Str Val Loss": round(str_val_loss.item(), 4),
                }

                ratio = client.num_nodes() / self.server.num_nodes()
                for key, val in result.items():
                    if key not in average_result.keys():
                        average_result[key] = ratio * val
                    else:
                        average_result[key] += ratio * val

                if epoch == epochs - 1:
                    self.LOGGER.info(f"SDWA results for client{client.id}:")
                    self.LOGGER.info(f"{result}")

            total_loss = loss_list.mean()
            total_loss.backward()

            for optimizer in optimizers.values():
                optimizer.step()

            with torch.no_grad():
                self.calc_server_weights(clients)

            average_result["Epoch"] = epoch + 1
            average_results.append(average_result)

            metrics[f"average train acc"] = average_result["Train Acc"]
            metrics[f"average val acc"] = average_result["Val Acc"]
            with torch.no_grad():
                metrics["Total Loss"] = round(total_loss.item(), 4)
                metrics["Structure Loss"] = round(structure_loss.mean().item(), 4)

            bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
            bar.set_postfix(metrics)
            bar.update()

        test_metrics = self.test(clients)
        test_acc = test_metrics[f"Client{self.server.id}"]["Test Acc"]
        self.LOGGER.info(f"Server test accuracy: {test_acc:0.4f}")

        average_test_acc = 0
        for client in clients:
            test_acc = test_metrics[f"Client{client.id}"]["Test Acc"]
            self.LOGGER.info(f"Client{client.id} test accuracy: {test_acc}")
            average_test_acc += test_acc * client.num_nodes() / self.server.num_nodes()

        self.LOGGER.info(f"Average test accuracy: {average_test_acc:0.4f}")

        title = f"Server SDWA GNN"
        plot_metrics(server_results, title=title, save_path=self.save_path)

        title = f"Average SDWA GNN"
        plot_metrics(average_results, title=title, save_path=self.save_path)

        if config.structure_model.structure_type == "mp":
            print(self.structure_model.mp_layer.state_dict()["weight"])

    def train_SDGA(self, clients, epochs=1):
        optimizer = self.create_SG_optimizer()
        metrics = {}
        server_results = []
        average_results = []

        bar = tqdm(total=epochs, position=0)
        for epoch in range(epochs):
            optimizer.zero_grad()

            (
                cls_train_loss,
                str_train_loss,
                train_acc,
                cls_val_loss,
                str_val_loss,
                val_acc,
            ) = self.step(train=False)

            result = {
                "Cls Train Loss": round(cls_train_loss.item(), 4),
                "Train Acc": round(train_acc, 4),
                "Cls Val Loss": round(cls_val_loss.item(), 4),
                "Val Acc": round(val_acc, 4),
                "Str Train Loss": round(str_train_loss.item(), 4),
                "Str Val Loss": round(str_val_loss.item(), 4),
                "Epoch": epoch,
            }

            server_results.append(result)
            metrics[f"server train acc"] = result["Train Acc"]
            metrics[f"server val acc"] = result["Val Acc"]
            if epoch == epochs - 1:
                self.LOGGER.info(f"SDGA results for client{self.server.id}:")
                self.LOGGER.info(f"{result}")

            self.server.zero_grad()
            server_weights = self.server.state_dict()
            for client in clients:
                client.zero_grad()
                client.load_state_dict(server_weights)

            out = self.model()
            structure_loss = self.calc_loss(out["structure_model"])
            loss_list = torch.zeros(len(clients), dtype=torch.float32)
            average_result = {}
            for ind, client in enumerate(clients):
                y_pred = out[f"client{client.id}"]
                (
                    cls_train_loss,
                    str_train_loss,
                    train_acc,
                    cls_val_loss,
                    str_val_loss,
                    val_acc,
                ) = self.cacl_metrics(client, y_pred, structure_loss)

                train_loss = cls_train_loss + str_train_loss
                loss_list[ind] = train_loss

                result = {
                    "Cls Train Loss": round(cls_train_loss.item(), 4),
                    "Train Acc": round(train_acc, 4),
                    "Cls Val Loss": round(cls_val_loss.item(), 4),
                    "Val Acc": round(val_acc, 4),
                    "Str Train Loss": round(str_train_loss.item(), 4),
                    "Str Val Loss": round(str_val_loss.item(), 4),
                }

                ratio = client.num_nodes() / self.server.num_nodes()
                for key, val in result.items():
                    if key not in average_result.keys():
                        average_result[key] = ratio * val
                    else:
                        average_result[key] += ratio * val

                if epoch == epochs - 1:
                    self.LOGGER.info(f"SDGA results for client{client.id}:")
                    self.LOGGER.info(f"{result}")

            total_loss = loss_list.mean()
            total_loss.backward()
            self.calc_total_grads(clients)

            optimizer.step()

            average_result["Epoch"] = epoch + 1
            average_results.append(average_result)

            metrics[f"average train acc"] = average_result["Train Acc"]
            metrics[f"average val acc"] = average_result["Val Acc"]

            with torch.no_grad():
                metrics["Total Loss"] = round(total_loss.item(), 4)
                metrics["Structure Loss"] = round(structure_loss.mean().item(), 4)

            bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
            bar.set_postfix(metrics)
            bar.update()

        test_metrics = self.test(clients)
        test_acc = test_metrics[f"Client{self.server.id}"]["Test Acc"]
        self.LOGGER.info(f"Server test accuracy: {test_acc:0.4f}")

        average_test_acc = 0
        for client in clients:
            test_acc = test_metrics[f"Client{client.id}"]["Test Acc"]
            self.LOGGER.info(f"Client{client.id} test accuracy: {test_acc}")
            average_test_acc += test_acc * client.num_nodes() / self.server.num_nodes()

        self.LOGGER.info(f"Average test accuracy: {average_test_acc:0.4f}")

        title = f"Server SDGA GNN"
        plot_metrics(server_results, title=title, save_path=self.save_path)

        title = f"Average SDGA GNN"
        plot_metrics(average_results, title=title, save_path=self.save_path)

        if config.structure_model.structure_type == "mp":
            print(self.structure_model.mp_layer.state_dict()["weight"])
