import logging
from itertools import product

import numpy as np
import torch
from torch.nn.functional import normalize
from torch_geometric.nn import MessagePassing
from tqdm import tqdm
from src.client import Client

from src.models.GNN_models import GNN, MLP, ModelBinder, ModelSpecs, calc_accuracy
from src.models.structure_model3 import JointModel
from src.utils.config_parser import Config
from src.utils.graph import Graph
from src.utils.utils import *

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
            [config.structure_model.GNN_structure_layers_sizes[-1]]
            + config.feature_model.mlp_layer_sizes
            + [num_classes],
            dropout=0.1,
            normalization="batch",
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

    def set_GNN_model(self, client_models, server_model):
        self.server: Client = server_model

        layer_sizes = [
            config.structure_model.num_structural_features
        ] + config.structure_model.GNN_structure_layers_sizes

        self.structure_model = GNN(
            layer_sizes=layer_sizes,
            last_layer="linear",
            layer_type=config.model.gnn_layer_type,
            dropout=config.model.dropout,
            normalization="batch",
            multiple_features=config.structure_model.structure_type == "mp",
            feature_dims=config.structure_model.num_mp_vectors,
        )

        self.model = JointModel(
            server=server_model,
            clients=client_models,
            logger=self.LOGGER,
        )

    def set_MP_model(self, client_models, server_model):
        self.server: Client = server_model

        if config.structure_model.structure_type == "random":
            layer_sizes = [config.structure_model.num_structural_features]
        else:
            layer_sizes = [
                config.structure_model.num_structural_features
            ] + config.structure_model.MP_structure_layers_sizes

        models_specs = [
            ModelSpecs(
                type="MLP",
                layer_sizes=layer_sizes,
                final_activation_function="linear",
                normalization="batch",
            ),
            ModelSpecs(
                type="MP",
                num_layers=config.structure_model.mp_layers,
            ),
        ]

        self.structure_model: ModelBinder = ModelBinder(models_specs)

        self.model = JointModel(
            server=server_model,
            clients=client_models,
            logger=self.LOGGER,
        )

    def initialize_structural_graph(
        self,
        client_models,
        server_model,
        propagate_type=config.model.propagate_type,
    ):
        self.prepare_data()
        if propagate_type == "GNN":
            self.set_GNN_model(
                client_models,
                server_model,
            )
        elif propagate_type == "MP":
            self.set_MP_model(
                client_models,
                server_model,
            )

    def reset_parameters(self):
        self.graph.reset_parameters()
        self.model.reset_parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, weights):
        self.model.load_state_dict(weights)

    def train(self, mode: bool = True):
        self.structure_model.train(mode)

    def eval(self):
        self.structure_model.eval()

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
            ).item()
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

    def get_structure_embeddings(self):
        x = self.graph.structural_features
        edge_index = self.graph.edge_index
        S = self.structure_model(x, edge_index)

        return S

    def fit(
        self,
        epochs=config.model.epoch_classifier,
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
            self.structure_model.train()
            S = self.get_structure_embeddings()
            loss = self.calc_loss(S)

            train_loss = loss[self.graph.train_mask].mean()
            val_loss = loss[self.graph.val_mask].mean()

            train_loss.backward()
            optimizer.step()

            if predict:
                self.structure_model.eval()
                with torch.no_grad():
                    x = self.get_structure_embeddings()
                x_train = x[self.graph.train_mask]
                y_train = self.y[self.graph.train_mask]
                x_val = x[self.graph.val_mask]
                y_val = self.y[self.graph.val_mask]

                # self.cls.reset_parameters()
                cls_val_acc, cls_val_loss = self.cls.fit(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    epochs=100,
                )
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
        self.structure_model.eval()
        x = self.get_structure_embeddings()
        y = self.y
        test_acc = self.cls.test(x[self.graph.test_mask], y[self.graph.test_mask])
        return test_acc

    @torch.no_grad()
    def test(self, clients):
        self.model.eval()
        metrics = {}

        y = self.server.subgraph.y
        test_mask = self.server.subgraph.test_mask
        y_pred = self.model.step(self.server)
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

    def cacl_metrics(self, client, y_pred, structure_loss=None):
        node_ids = client.get_nodes()
        y = client.subgraph.y
        train_mask = client.subgraph.train_mask
        val_mask = client.subgraph.val_mask

        if structure_loss is not None:
            client_structure_loss = structure_loss[node_ids]

        cls_train_loss = self.criterion(y_pred[train_mask], y[train_mask])
        if structure_loss is not None:
            str_train_loss = client_structure_loss[train_mask].mean()
        else:
            str_train_loss = torch.tensor(0)

        train_acc = calc_accuracy(
            y_pred[train_mask].argmax(dim=1),
            y[train_mask],
        )

        # Validation
        with torch.no_grad():
            cls_val_loss = self.criterion(y_pred[val_mask], y[val_mask])
            if structure_loss is not None:
                str_val_loss = client_structure_loss[val_mask].mean()
            else:
                str_val_loss = torch.tensor(0)

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
        self.model.train(train)
        S = self.get_structure_embeddings()
        h = self.server.get_feature_embeddings()
        x = torch.hstack((h, S))
        y_pred = self.server.predict_labels(x)

        if config.structure_model.sd_ratio != 0:
            structure_loss = self.calc_loss(S)
        else:
            structure_loss = None

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
        client: Client
        for client in self.model:
            parameters = client.parameters()
            # parameters += list(self.models[f"structure_model"].parameters())
            optimizer = torch.optim.Adam(
                parameters, lr=config.model.lr, weight_decay=config.model.weight_decay
            )
            optimizers[f"client{client.id}"] = optimizer

        parameters = list(self.structure_model.parameters())
        if config.structure_model.structure_type == "random":
            parameters += [self.graph.structural_features]

        if len(parameters) > 0:
            optimizer = torch.optim.Adam(
                parameters, lr=config.model.lr, weight_decay=config.model.weight_decay
            )
            optimizers[f"structure_model"] = optimizer

        return optimizers

    def create_SG_optimizer(self):
        parameters = list(self.server.parameters()) + list(
            self.structure_model.parameters()
        )

        if config.structure_model.structure_type == "random":
            parameters += [self.graph.structural_features]

        optimizer = torch.optim.Adam(
            parameters, lr=config.model.lr, weight_decay=config.model.weight_decay
        )

        return optimizer

    def reset_optimizers(optimizers) -> None:
        for optimizer in optimizers.values():
            optimizer.zero_grad()

    def train_SD_Server(
        self,
        epochs=config.model.epoch_classifier,
        log=True,
        plot=True,
    ):
        final_result = {}
        optimizer = self.create_SG_optimizer()
        metrics = {}
        plot_results = {}
        plot_results[f"Client{self.server.id}"] = []

        if log:
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

            if log:
                metrics["Val ACC"] = result["Val Acc"]
                metrics["Cls Val Loss"] = result["Cls Val Loss"]
                metrics["Str Val Loss"] = result["Str Val Loss"]
                if epoch == epochs - 1:
                    self.LOGGER.info(f"SD_Server results for client{self.server.id}:")
                    self.LOGGER.info(f"{result}")

            train_loss = (
                cls_train_loss + config.structure_model.sd_ratio * str_train_loss
            )

            train_loss.backward()
            optimizer.step()

            if log:
                bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
                bar.set_postfix(metrics)
                bar.update()

        test_metrics = self.test([])
        test_acc = test_metrics[f"Client{self.server.id}"]["Test Acc"]
        final_result[f"Server"] = test_acc
        if log:
            self.LOGGER.info(f"Server test accuracy: {test_acc:0.4f}")

        if plot:
            title = f"Central SDSFL GNN"
            plot_metrics(
                plot_results[f"Client{self.server.id}"],
                title=title,
                save_path=self.save_path,
            )

        if config.structure_model.structure_type == "mp":
            print(self.structure_model.mp_layer.state_dict()["weight"])

        return final_result

    def train_SDWA(
        self,
        clients,
        epochs=config.model.epoch_classifier,
        log=True,
        plot=True,
    ):
        final_result = {}
        metrics = {}
        optimizers = self.create_SW_optimizers()
        server_results = []
        average_results = []

        if log:
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

            if log:
                metrics[f"server train acc"] = result["Train Acc"]
                metrics[f"server val acc"] = result["Val Acc"]
                if epoch == epochs - 1:
                    self.LOGGER.info(f"SDWA results for {self.server.id}:")
                    self.LOGGER.info(f"{result}")

            self.server.zero_grad()
            server_weights = self.server.state_dict()
            for client in clients:
                client.load_state_dict(server_weights)
            self.model.train()
            out = self.model()
            if config.structure_model.sd_ratio != 0:
                structure_loss = self.calc_loss(out["structure_model"])
            else:
                structure_loss = None

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

                train_loss = (
                    cls_train_loss + config.structure_model.sd_ratio * str_train_loss
                )
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

                if log:
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

            if log:
                metrics[f"average train acc"] = average_result["Train Acc"]
                metrics[f"average val acc"] = average_result["Val Acc"]
                with torch.no_grad():
                    metrics["Total Loss"] = round(total_loss.item(), 4)
                    if structure_loss is not None:
                        metrics["Structure Loss"] = round(
                            structure_loss.mean().item(), 4
                        )

                bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
                bar.set_postfix(metrics)
                bar.update()

        test_metrics = self.test(clients)
        test_acc = test_metrics[f"Client{self.server.id}"]["Test Acc"]
        final_result["Server"] = test_acc
        if log:
            self.LOGGER.info(f"Server test accuracy: {test_acc:0.4f}")

        average_test_acc = 0
        for client in clients:
            test_acc = test_metrics[f"Client{client.id}"]["Test Acc"]
            final_result[f"Client{client.id}"] = test_acc
            if log:
                self.LOGGER.info(f"Client{client.id} test accuracy: {test_acc}")
            average_test_acc += test_acc * client.num_nodes() / self.server.num_nodes()

        final_result["Average"] = average_test_acc
        if log:
            self.LOGGER.info(f"Average test accuracy: {average_test_acc:0.4f}")

        if plot:
            title = f"Server SDWA GNN"
            plot_metrics(server_results, title=title, save_path=self.save_path)

            title = f"Average SDWA GNN"
            plot_metrics(average_results, title=title, save_path=self.save_path)

        if config.structure_model.structure_type == "mp":
            print(self.structure_model.mp_layer.state_dict()["weight"])

        return final_result

    def train_SDGA(
        self,
        clients,
        epochs=config.model.epoch_classifier,
        log=True,
        plot=True,
    ):
        final_result = {}
        metrics = {}
        optimizer = self.create_SG_optimizer()
        server_results = []
        average_results = []

        if log:
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
            if log:
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
            self.model.train()
            out = self.model()
            if config.structure_model.sd_ratio != 0:
                structure_loss = self.calc_loss(out["structure_model"])
            else:
                structure_loss = None
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

                train_loss = (
                    cls_train_loss + config.structure_model.sd_ratio * str_train_loss
                )
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
                if log:
                    if epoch == epochs - 1:
                        self.LOGGER.info(f"SDGA results for client{client.id}:")
                        self.LOGGER.info(f"{result}")

            total_loss = loss_list.mean()
            total_loss.backward()
            self.calc_total_grads(clients)

            optimizer.step()

            average_result["Epoch"] = epoch + 1
            average_results.append(average_result)

            if log:
                metrics[f"average train acc"] = average_result["Train Acc"]
                metrics[f"average val acc"] = average_result["Val Acc"]

                with torch.no_grad():
                    metrics["Total Loss"] = round(total_loss.item(), 4)
                    if structure_loss is not None:
                        metrics["Structure Loss"] = round(
                            structure_loss.mean().item(), 4
                        )

                bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
                bar.set_postfix(metrics)
                bar.update()

        test_metrics = self.test(clients)
        test_acc = test_metrics[f"Client{self.server.id}"]["Test Acc"]
        final_result["Server"] = test_acc
        if log:
            self.LOGGER.info(f"Server test accuracy: {test_acc:0.4f}")

        average_test_acc = 0
        for client in clients:
            test_acc = test_metrics[f"Client{client.id}"]["Test Acc"]
            final_result[f"Client{client.id}"] = test_acc
            if log:
                self.LOGGER.info(f"Client{client.id} test accuracy: {test_acc}")
            average_test_acc += test_acc * client.num_nodes() / self.server.num_nodes()

        final_result["Average"] = average_test_acc
        if log:
            self.LOGGER.info(f"Average test accuracy: {average_test_acc:0.4f}")

        if plot:
            title = f"Server SDGA GNN"
            plot_metrics(server_results, title=title, save_path=self.save_path)

            title = f"Average SDGA GNN"
            plot_metrics(average_results, title=title, save_path=self.save_path)

        if config.structure_model.structure_type == "mp":
            print(self.structure_model.mp_layer.state_dict()["weight"])

        return final_result

    def train_local_sd(
        self,
        clients,
        epochs=config.model.epoch_classifier,
        log=True,
        plot=True,
    ):
        optimizers = self.create_SW_optimizers()
        metrics = {}
        average_results = []
        # sd_ratios = torch.randn((len(clients)), requires_grad=True)

        client: Client
        # for ind, client in enumerate(clients):
        #     optimizers[f"client{client.id}"].param_groups[0]["params"].append(
        #         sd_ratios[ind]
        #     )

        if log:
            bar = tqdm(total=epochs, position=0)
        for epoch in range(epochs):
            StructurePredictor.reset_optimizers(optimizers)
            self.model.train()
            out = self.model()
            if config.structure_model.sd_ratio != 0:
                structure_loss = self.calc_loss(out["structure_model"])
            else:
                structure_loss = None

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

                train_loss = (
                    # cls_train_loss
                    # + sd_ratios[ind] * str_train_loss
                    cls_train_loss
                    + config.structure_model.sd_ratio * str_train_loss
                )
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

                if log:
                    if epoch == epochs - 1:
                        self.LOGGER.info(f"local SD results for client{client.id}:")
                        self.LOGGER.info(f"{result}")

            total_loss = loss_list.mean()
            total_loss.backward()

            for optimizer in optimizers.values():
                optimizer.step()

            average_result["Epoch"] = epoch + 1
            average_results.append(average_result)

            if log:
                metrics[f"average train acc"] = average_result["Train Acc"]
                metrics[f"average val acc"] = average_result["Val Acc"]
                with torch.no_grad():
                    metrics["Total Loss"] = round(total_loss.item(), 4)
                    if structure_loss is not None:
                        metrics["Structure Loss"] = round(
                            structure_loss.mean().item(), 4
                        )

                bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
                bar.set_postfix(metrics)
                bar.update()

        test_metrics = self.test(clients)

        average_test_acc = 0
        for client in clients:
            test_acc = test_metrics[f"Client{client.id}"]["Test Acc"]
            if log:
                self.LOGGER.info(f"Client{client.id} test accuracy: {test_acc}")
            average_test_acc += test_acc * client.num_nodes() / self.server.num_nodes()

        if log:
            self.LOGGER.info(f"Average test accuracy: {average_test_acc:0.4f}")

        if plot:
            title = f"Average local SD GNN"
            plot_metrics(average_results, title=title, save_path=self.save_path)

        if config.structure_model.structure_type == "mp":
            print(self.structure_model.mp_layer.state_dict()["weight"])

        # print(f"sd ratios: {sd_ratios}")
