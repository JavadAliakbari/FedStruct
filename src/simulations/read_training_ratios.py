from os import listdir
import os
import sys

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # folder_path = "results/Paper Results/Cora/louvain-10/0.1/"
    # dataset = "Chameleon"
    # partioning = "kmeans"
    dataset = "Cora"
    partioning = "random"
    num_subgraphs = 10
    plot_data = []

    models = [
        "server_GNN_true",
        "local_GNN_true",
        "flga_GNN_true",
        # "flga_node2vec_DGCN_true",
        "flga_fedstar_DGCN_true",
        "flga_fedstar_DGCN_prune",
        "flga_hop2vec_DGCN_true",
        "flga_hop2vec_DGCN_prune",
        "flga_degree_DGCN_true",
        "flga_degree_DGCN_prune",
        # "flga_DGCN",
        # "hop2vec_sdga_GNN",
        # "flga_GNN",
        "fedsage+_GA",
        "fedpub",
        # "local_GNN",
        # "server_mlp",
        # "flga_mlp",
        # "local_mlp",
    ]

    train_ratio_list = [
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        # 0.5,
        # 0.55,
        # 0.6,
    ]
    base_path = f"results/Training/{dataset}/{partioning}/{num_subgraphs}/"
    for train_ratio in train_ratio_list:
        folder_path = f"{base_path}{train_ratio}/"
        filenames = listdir(folder_path)
        filename = [filename for filename in filenames if filename.endswith(".csv")][0]
        # folder_path = f"ICML Results/train_ratio/{dataset}/{partioning}/{num_subgraphs}/{train_ratio}/"
        path = f"{folder_path}{filename}"
        df = pd.read_csv(path, index_col="Unnamed: 0")

        df2 = df.loc[models, "0"].tolist()
        data = [x.split("±") for x in df2]
        data = [100 * float(x[0]) for x in data]
        plot_data.append(data)
        # data = [rf"{x[0]:0.2f}$\pm$ {x[1]:0.2f}" for x in data]
        # with open(f"{folder_path}paper_DGCN.txt", "w") as fid:
        #     for line in data:
        #         fid.write(f"{line}\n")

    # models.append("server GNN")
    # plot_data = np.array(plot_data)
    # a = np.array([78.64, 82.13, 84.23, 85.08, 85.69, 86.07, 86.41, 86.8, 87.36, 87.46])
    # np.append(plot_data, a, axis=0)
    # plot_data = np.concatenate((plot_data, a[:, np.newaxis]), axis=1)
    df = pd.DataFrame(plot_data, columns=models, index=train_ratio_list)
    df.to_csv(f"{base_path}{dataset}_ratio.csv")

    plt.plot(train_ratio_list, plot_data, "-*", label=models)
    plt.legend()
    plt.xlabel("training ratio")
    plt.ylabel("accuracy")
    plt.title(f"training ratio vs accuracy for {dataset} {partioning} partitioning")
    plt.show()

# server_GNN
