from os import listdir
import os
import sys

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

import pandas as pd

if __name__ == "__main__":
    # folder_path = "results/Paper Results/Cora/louvain-10/0.1/"
    training_ratio = 0.1
    datas = []
    base_path = f"results/Neurips/Chalmers/"
    for dataset in [
        "Cora",
        "CiteSeer",
        "PubMed",
        # "Chameleon",
        # "Photo",
        # "Amazon-ratings",
    ]:
        for partioning in [
            "louvain",
            "random",
            "kmeans",
        ]:
            # if partioning == "random":
            #     num_subgraphs_list = [5, 10, 20]
            # else:
            #     num_subgraphs_list = [10]
            num_subgraphs_list = [10]
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
                    # "flga_hop2vec_DGCN_true",
                    # "flga_GDV_DGCN_true",
                    # "flga_node2vec_DGCN_true",
                    # "flga_hop2vec_GNN_true",
                    # "flga_GDV_GNN_true",
                    # "flga_node2vec_GNN_true",
                    "flwa_MLP",
                    "flwa_GNN_true",
                    "flwa_DGCN_true",
                ]

                df2 = df.loc[rows, "0"].tolist()
                data = [x.split("±") for x in df2]
                data = [[100 * float(x[0]), 100 * float(x[1])] for x in data]
                data = [rf"{x[0]:0.2f}$\pm$ {x[1]:0.2f}" for x in data]
                datas.append(data)

    with open(f"{base_path}paper_DGCN.txt", "w") as fid:
        for i in range(len(rows)):
            s = ""
            for line in datas:
                s += f"& {line[i]} "
            fid.write(f"{s} \\\\ \n")
    a = 1
