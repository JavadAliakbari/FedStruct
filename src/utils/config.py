import torch

root_path = "/Users/javada/Documents/Ph.D Projects/GNN + FL/AML/fedsage/"

seed = 2021
no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()

# dataset = "PubMed"
dataset = "Cora"
# dataset = "squirrel"
# dataset = "chameleon"
# dataset = "Minesweeper"
# dataset = "Tolokers"
# dataset = "Toy Example"
num_subgraphs = 3
num_owners = num_subgraphs
delta = 20

num_samples = [5, 5]
batch_size = 64
latent_dim = 128
steps = 10
epochs_local = 1
lr = 0.001
weight_decay = 1e-4
hidden = 32
dropout = 0.25

gen_epochs = 100
sd_ratio = 1

epoch_classifier = 150
classifier_layer_sizes = [512, 256]
structure_layers_size = [128, 64]

a = 1
b = 1
c = 1

structure_type = "degree"
# structure_type = "GDV"
# structure_type = "node2vec"
# num_structural_features = 100
num_structural_features = 73
structure_dim_out = 64
