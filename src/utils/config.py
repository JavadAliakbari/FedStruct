import torch

root_path = "/Users/javada/Documents/Ph.D Projects/GNN + FL/AML/fedsage/"

seed = 2021
no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()

# dataset = "PubMed"
dataset = "Cora"
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
dropout = 0.5

gen_epochs = 100
num_pred = 5
hidden_portion = 0.5

epoch_classifier = 200
classifier_layer_sizes = [64, 32]
structure_layers_size = [64, 64]

a = 1
b = 1
c = 1

num_structural_features = 100
structure_dim_out = 64
