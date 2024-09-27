from os import listdir
import os
import sys

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # folder_path = "results/Paper Results/Cora/louvain-10/0.1/"
    training_ratio = 0.1
    datas = []
    datas2 = []
    base_path = f"results/Clients/"
    for dataset in [
        # "Cora",
        # "CiteSeer",
        # "PubMed",
        "chameleon",
        # "Photo",
        # "Amazon-ratings",
    ]:
        for partioning in [
            # "louvain",
            "random",
            # "kmeans",
        ]:
            # if partioning == "random":
            #     num_subgraphs_list = [5, 10, 20]
            # else:
            #     num_subgraphs_list = [10]
            # num_subgraphs_list = [10]
            num_subgraphs_list = np.arange(5, 45, 5)
            for num_subgraphs in num_subgraphs_list:
                folder_path = f"{base_path}{dataset}/{partioning}/{num_subgraphs}/{training_ratio}/"
                # folder_path = f"ICML Results/Paper Results48/{dataset}/{partioning}/{num_subgraphs}/0.1/"
                filenames = listdir(folder_path)
                filename = [
                    filename for filename in filenames if filename.endswith(".csv")
                ][0]
                # folder_path = f"ICML Results/train_ratio/{dataset}/{partioning}/{num_subgraphs}/{train_ratio}/"
                path = f"{folder_path}{filename}"
                df = pd.read_csv(path, index_col="Unnamed: 0")
                rows = [
                    "flga_hop2vec_DGCN_prune",
                    # "flga_hop2vec_DGCN_true",
                    # "flga_GDV_DGCN_true",
                    # "flga_node2vec_DGCN_true",
                    # "flga_hop2vec_GNN_true",
                    # "flga_GDV_GNN_true",
                    # "flga_node2vec_GNN_true",
                    "flga_MLP",
                    "server_MLP",
                    "fedpub",
                    "fedgcn1",
                    "fedgcn2",
                    # "flwa_GNN_true",
                    "flga_DGCN_prune",
                    # "flwa_DGCN_true",
                ]

                df2 = df.loc[rows, "0"].tolist()
                data_ = [x.split("±") for x in df2]
                data = [float(x[0]) for x in data_]
                data2 = [float(x[1]) for x in data_]
                # data = [rf"{x[0]:0.2f}$\pm$ {x[1]:0.2f}" for x in data]
                datas.append(data)
                datas2.append(data2)

    datas = np.array(datas)
    datas2 = np.array(datas2)
    plt.plot(num_subgraphs_list, datas, "-*", label=rows)
    for data1, data2 in zip(datas.T, datas2.T):
        plt.fill_between(num_subgraphs_list, data1 - data2, data1 + data2, alpha=0.2)
    plt.title("Number of Clients vs Accuracy")
    plt.xlabel("Num of Clients")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
