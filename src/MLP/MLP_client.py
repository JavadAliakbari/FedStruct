import os

from src.utils.config_parser import Config
from src.utils.graph import Data
from src.client import Client
from src.MLP.MLP_classifier import MLPClassifier

path = os.environ.get("CONFIG_PATH")
config = Config(path)


class MLPClient(Client):
    def __init__(
        self,
        graph: Data,
        id: int = 0,
        save_path="./",
        logger=None,
    ):
        super().__init__(
            graph=graph,
            id=id,
            classifier_type="MLP",
            save_path=save_path,
            logger=logger,
        )

        self.classifier: MLPClassifier = MLPClassifier(
            id=self.id,
            save_path=self.save_path,
            logger=self.LOGGER,
        )

    def initialize(
        self,
        dim_in=None,
    ):
        self.classifier.restart()
        self.classifier.prepare_data(
            data=self.graph,
            batch_size=config.model.batch_size,
        )
        self.classifier.set_classifiers(dim_in=dim_in)

    def train_local_model(
        self,
        epochs=config.model.epoch_classifier,
        log=True,
        plot=True,
    ):
        self.initialize()
        return super().train_local_model(
            epochs=epochs,
            log=log,
            plot=plot,
            model_type="MLP",
        )
