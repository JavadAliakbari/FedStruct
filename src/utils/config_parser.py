import yaml


class Config:
    def __init__(self, path="./config/config.yml"):
        self.config = Config.load_config(path)

        self.dataset = Dataset(self.config["dataset"])
        self.subgraph = Subgraph(self.config["subgraph"])
        self.model = Model(self.config["model"])
        self.structure_model = StructureModel(self.config["structure_model"])

    def load_config(path):
        with open(path) as f:
            config = yaml.load(f, yaml.FullLoader)

        return config


class Dataset:
    def __init__(self, dataset):
        self.load_config(dataset)

    def load_config(self, dataset):
        self.dataset_name = dataset["dataset_name"]


class Subgraph:
    def __init__(self, subgraph):
        self.load_config(subgraph)

    def load_config(self, subgraph):
        self.num_subgraphs = subgraph["num_subgraphs"]
        self.delta = subgraph["delta"]


class Model:
    def __init__(self, model):
        self.load_config(model)

    def load_config(self, model):
        self.num_samples = model["num_samples"]
        self.batch_size = model["batch_size"]
        self.latent_dim = model["latent_dim"]
        self.steps = model["steps"]
        self.epochs_local = model["epochs_local"]
        self.lr = model["lr"]
        self.weight_decay = model["weight_decay"]
        self.hidden = model["hidden"]
        self.dropout = model["dropout"]
        self.gen_epochs = model["gen_epochs"]
        self.epoch_classifier = model["epoch_classifier"]
        self.classifier_layer_sizes = model["classifier_layer_sizes"]


class StructureModel:
    def __init__(self, structure_model):
        self.load_config(structure_model)

    def load_config(self, structure_model):
        self.sd_ratio = structure_model["sd_ratio"]
        self.structure_layers_size = structure_model["structure_layers_size"]
        self.structure_type = structure_model["structure_type"]
        self.num_structural_features = structure_model["num_structural_features"]
