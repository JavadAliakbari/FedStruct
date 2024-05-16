import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # folder_path = "results/Paper Results/Cora/louvain-10/0.1/"
    dataset = "Chameleon"
    # dataset = "Cora"
    # partioning = "random"
    partioning = "kmeans"
    num_subgraphs = 5
    plot_data = []

    models = [
        "server_GNN",
        "node2vec_sdga_DGCN",
        "random_sdga_DGCN",
        # "node2vec_sdga_GNN",
        "degree_sdga_DGCN",
        # "flga_DGCN",
        # "random_sdga_GNN",
        "flga_GNN",
        "fedsage+_GA_GNN",
        "local_GNN",
        # "server_mlp",
        # "flga_mlp",
        # "local_mlp",
    ]

    train_ratio_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for train_ratio in train_ratio_list:
        folder_path = f"ICML Results/train_ratio/{dataset}/{partioning}/{num_subgraphs}/{train_ratio}/"
        # folder_path = f"ICML Results/Paper Results48/{dataset}/{partioning}/{num_subgraphs}/0.1/"
        # folder_path = f"results/Paper Results2/{dataset}/{partioning}/{num_subgraphs}/0.1/"
        path = f"{folder_path}final_result_{train_ratio}.csv"
        df = pd.read_csv(path, index_col="Unnamed: 0")

        df2 = df.loc[models, "0"].tolist()
        data = [x.split("±") for x in df2]
        data = [100 * float(x[0]) for x in data]
        plot_data.append(data)
        # data = [rf"{x[0]:0.2f}$\pm$ {x[1]:0.2f}" for x in data]
        # with open(f"{folder_path}paper_DGCN.txt", "w") as fid:
        #     for line in data:
        #         fid.write(f"{line}\n")

plot_data = np.array(plot_data)

plt.plot(train_ratio_list, plot_data, label=models)
plt.legend()
plt.xlabel("training ratio")
plt.ylabel("accuracy")
plt.title(f"training ratio vs accuracy for {dataset} {partioning} partitioning")
plt.show()

a = 1
