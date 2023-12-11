from ast import List

import torch
from tqdm import tqdm
from src.GNN_client import GNNClient

from src.utils.utils import *
from src.utils.graph import Graph
from src.utils.config_parser import Config
from src.server import Server
from src.GNN_classifier import GNNClassifier

config = Config()


class GNNServer(Server, GNNClient):
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

        self.clients: List[GNNClient] = []

    def add_client(self, subgraph):
        client = GNNClient(
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
        structure=False,
    ) -> None:
        self.initialize(
            propagate_type=propagate_type,
            structure=structure,
        )
        client: GNNClient
        for client in self.clients:
            client.initialize(
                propagate_type=propagate_type,
                structure=structure,
                get_structure_embeddings=self.get_structure_embeddings2,
            )

        if structure:
            self.graph.add_structural_features(
                structure_type=config.structure_model.structure_type,
                num_structural_features=config.structure_model.num_structural_features,
            )

            self.set_SFV(self.graph.structural_features)

            if propagate_type == "MP":
                abar = self.obtain_a()

                self.share_abar(abar)
                self.set_abar(abar)

                self.share_SFV()

        self.initialized = True

    def obtain_a(self):
        if config.structure_model.estimate:
            abar = estimate_a(
                self.graph.edge_index,
                self.graph.num_nodes,
                config.structure_model.mp_layers,
            )
        else:
            abar = obtain_a(
                self.graph.edge_index,
                self.graph.num_nodes,
                config.structure_model.mp_layers,
            )

        # abar_ = abar.to_dense().numpy()
        # abar1_ = abar1.to_dense().numpy()
        # e = np.mean(np.abs(abar_ - abar1_) ** 2)
        # print(e)

        return abar

    def share_abar(self, abar):
        num_nodes = self.graph.num_nodes
        row, col, val = abar.coo()

        client: GNNClient
        for client in self.clients:
            nodes = client.get_nodes()
            node_map = client.graph.node_map

            cond = torch.isin(row, nodes)
            row_i = row[cond]
            row_i = torch.tensor(itemgetter(*np.array(row_i))(node_map))
            col_i = col[cond]
            val_i = val[cond]
            abar_i = SparseTensor(
                row=row_i,
                col=col_i,
                value=val_i,
                sparse_sizes=(len(nodes), num_nodes),
            )

            client.set_abar(abar_i)

    def create_SFV(self):
        pass

    def share_SFV(self):
        SFV = self.graph.structural_features

        client: GNNClient
        for client in self.clients:
            client.set_SFV(SFV)

    def joint_train_g(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        log=True,
        plot=True,
        structure=False,
        FL=True,
    ):
        self.initialize_FL(propagate_type=propagate_type, structure=structure)

        if FL & structure:
            model_type = "SDGA"
        elif FL and not structure:
            model_type = "FLGA GNN"
        elif not FL and structure:
            model_type = "LocaL SDGA"
        else:
            model_type = "Local GNN"

        if propagate_type == "MP":
            model_type += "_MP"

        return super().joint_train_g(
            epochs=epochs, log=log, plot=plot, FL=FL, model_type=model_type
        )

    def joint_train_w(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        log=True,
        plot=True,
        structure=False,
        FL=True,
    ):
        self.initialize_FL(propagate_type=propagate_type, structure=structure)

        if FL & structure:
            model_type = "SDWA"
        elif FL and not structure:
            model_type = "FLWA GNN"
        elif not FL and structure:
            model_type = "LocaL SDWA"
        else:
            model_type = "Local GNN"

        if propagate_type == "MP":
            model_type += "_MP"

        return super().joint_train_w(
            epochs=epochs, log=log, plot=plot, FL=FL, model_type=model_type
        )

    def train_locsages(self, log=True, plot=True):
        client: GNNClient
        for client in self.clients:
            self.LOGGER.info(f"locsage for client{client.id}")
            client.train_locsage(log=log, plot=plot)
            self.LOGGER.info(
                f"Client{client.id} test accuracy: {client.test_classifier()}"
            )

    def train_fedgen(self):
        client: GNNClient
        other_client: GNNClient
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

        client: GNNClient
        other_client: GNNClient
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
            (train_loss, train_acc, _, val_loss, val_acc, _) = GNNClassifier.step(
                x=self.graph.x,
                y=self.graph.y,
                edge_index=self.graph.edge_index,
                model=self.classifier.feature_model,
                criterion=criterion,
                train_mask=self.graph.train_mask,
                val_mask=self.graph.val_mask,
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

        self.LOGGER.info(f"Server test accuracy: {self.test_classifier():0.4f}")

        average_test_acc = 0
        for client in self.clients:
            test_acc = client.test_classifier()
            self.LOGGER.info(f"Clinet{client.id} test accuracy: {test_acc:0.4f}")

            average_test_acc += test_acc * client.num_nodes() / self.num_nodes()
        self.LOGGER.info(f"Average test accuracy: {average_test_acc:0.4f}")

        title = f"Server fedsage+ {self.classifier_type}"
        plot_metrics(server_results, title=title, save_path=self.save_path)

        title = f"Average fedsage+ {self.classifier_type}"
        plot_metrics(average_results, title=title, save_path=self.save_path)
