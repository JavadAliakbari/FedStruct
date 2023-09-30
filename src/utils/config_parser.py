import yaml


class Config:
    def __init__(self, path="config/config.yml"):
        self.config = Config.load_config(path)

        self.dataset = DatasetConfig(self.config["dataset"])
        self.subgraph = SubgraphConfig(self.config["subgraph"])
        self.model = ModelConfig(self.config["model"])
        self.feature_model = FeatureModelConfig(self.config["feature_model"])
        self.structure_model = StructureModelConfig(self.config["structure_model"])
        self.node2vec = Node2VecConfig(self.config["node2vec"])

    def load_config(path):
        with open(path) as f:
            config = yaml.load(f, yaml.FullLoader)

        return config


class DatasetConfig:
    def __init__(self, dataset):
        self.load_config(dataset)

    def load_config(self, dataset):
        self.dataset_name = dataset["dataset_name"]


class SubgraphConfig:
    def __init__(self, subgraph):
        self.load_config(subgraph)

    def load_config(self, subgraph):
        self.num_subgraphs = subgraph["num_subgraphs"]
        self.delta = subgraph["delta"]


class ModelConfig:
    def __init__(self, model):
        self.load_config(model)

    def load_config(self, model):
        self.num_samples = model["num_samples"]
        self.batch = model["batch"]
        self.batch_size = model["batch_size"]
        self.epochs_local = model["epochs_local"]
        self.lr = model["lr"]
        self.weight_decay = model["weight_decay"]
        self.gnn_layer_type = model["gnn_layer_type"]
        self.propagate_type = model["propagate_type"]
        self.dropout = model["dropout"]
        self.epoch_classifier = model["epoch_classifier"]
        self.metric = model["metric"]


class FeatureModelConfig:
    def __init__(self, feature_model):
        self.load_config(feature_model)

    def load_config(self, feature_model):
        self.gnn_layer_sizes = feature_model["gnn_layer_sizes"]
        self.mlp_layer_sizes = feature_model["mlp_layer_sizes"]
        self.mp_layers = feature_model["mp_layers"]


class StructureModelConfig:
    def __init__(self, structure_model):
        self.load_config(structure_model)

    def load_config(self, structure_model):
        self.sd_ratio = structure_model["sd_ratio"]
        self.GNN_structure_layers_sizes = structure_model["GNN_structure_layers_sizes"]
        self.MP_structure_layers_sizes = structure_model["MP_structure_layers_size"]
        self.mp_layers = structure_model["mp_layers"]
        self.structure_type = structure_model["structure_type"]
        self.num_structural_features = structure_model["num_structural_features"]
        self.loss = structure_model["loss"]
        self.num_mp_vectors = structure_model["num_mp_vectors"]
        self.cosine_similarity_predictor_epochs = structure_model[
            "cosine_similarity_predictor_epochs"
        ]
        self.gnn_epochs = structure_model["gnn_epochs"]
        self.mlp_epochs = structure_model["mlp_epochs"]


class Node2VecConfig:
    def __init__(self, node2vec):
        self.load_config(node2vec)

    def load_config(self, node2vec):
        self.epochs = node2vec["epochs"]
        self.walk_length = node2vec["walk_length"]
        self.context_size = node2vec["context_size"]
        self.walks_per_node = node2vec["walks_per_node"]
        self.num_negative_samples = node2vec["num_negative_samples"]
        self.p = node2vec["p"]
        self.q = node2vec["q"]
        self.show_bar = node2vec["show_bar"]
