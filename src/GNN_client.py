import os

from src.utils.utils import *
from src.utils.config_parser import Config
from src.client import Client
from src.utils.graph import Graph
from src.GNN_classifier import GNNClassifier

path = os.environ.get("CONFIG_PATH")
config = Config(path)


class GNNClient(Client):
    def __init__(
        self,
        graph: Graph,
        num_classes,
        id: int = 0,
        save_path="./",
        logger=None,
    ):
        super().__init__(
            graph=graph,
            num_classes=num_classes,
            id=id,
            classifier_type="GNN",
            save_path=save_path,
            logger=logger,
        )

        # self.LOGGER.info(f"Number of edges: {self.graph.num_edges}")

        self.classifier: GNNClassifier = GNNClassifier(
            id=self.id,
            num_classes=self.num_classes,
            save_path=self.save_path,
            logger=self.LOGGER,
        )

    def get_structure_embeddings2(self, node_ids):
        return self.classifier.get_structure_embeddings2(node_ids)

    def initialize(
        self,
        propagate_type=config.model.propagate_type,
        num_input_features=None,
        structure=False,
        structure_type=config.structure_model.structure_type,
        get_structure_embeddings=None,
    ) -> None:
        self.classifier.restart()
        self.classifier.prepare_data(
            graph=self.graph,
            batch_size=config.model.batch_size,
            num_neighbors=config.model.num_samples,
        )

        if propagate_type == "GNN":
            self.classifier.set_GNN_FPM(dim_in=num_input_features)
        elif propagate_type == "DGCN":
            self.classifier.set_DGCN_FPM(dim_in=num_input_features)

        if structure:
            if structure_type != "GDV":
                dim_in = config.structure_model.num_structural_features
            else:
                dim_in = 73

            if propagate_type == "GNN":
                self.classifier.set_GNN_SPM(
                    dim_in=dim_in,
                    get_structure_embeddings=get_structure_embeddings,
                )
            elif propagate_type == "DGCN":
                self.classifier.set_DGCN_SPM(dim_in=dim_in)

    def train_local_model(
        self,
        epochs=config.model.epoch_classifier,
        propagate_type=config.model.propagate_type,
        log=True,
        plot=True,
        structure=False,
    ):
        self.initialize(propagate_type=propagate_type, structure=structure)
        return super().train_local_model(
            epochs=epochs,
            log=log,
            plot=plot,
            model_type="GNN",
        )

    def set_abar(self, abar):
        self.classifier.set_abar(abar)

    def set_SFV(self, SFV):
        self.classifier.set_SFV(SFV)
