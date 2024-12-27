import os
import sys


pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

import pandas as pd
import matplotlib.pyplot as plt


def get_DF(path, partitioning, file_name="final_result_S", file_name_=None):
    # layers = range(1, 51)
    layers = list(range(1, 11)) + list(range(12, 21, 2)) + [23, 26, 30, 35, 40, 45, 50]

    data = {}
    for num_subgraphs in [5]:
        for train_ratio in [0.1]:
            keys = None
            for DGCN_layers in layers:
                try:
                    file_path = f"{path}{partitioning}/{num_subgraphs}/{train_ratio}/{DGCN_layers}/{file_name}.csv"
                    d = pd.read_csv(file_path).to_dict()
                except:
                    file_path_ = f"{path}{partitioning}/{num_subgraphs}/{train_ratio}/{DGCN_layers}/{file_name_}.csv"
                    d = pd.read_csv(file_path_).to_dict()
                keys = d["Unnamed: 0"].values()
                vals = d["0"].values()
                vals = [100 * float(val.split("\u00B1")[0]) for val in vals]
                data[DGCN_layers] = vals

    # print(data)
    final_path = f"{path}/{partitioning}/{num_subgraphs}/{train_ratio}/{file_name}.csv"
    df = pd.DataFrame(data, index=keys, columns=layers)
    return df, final_path


if __name__ == "__main__":
    # path = "results/Paper Results layers/Cora_100/"
    dataset_name = "Cora"
    path = f"results/Neurips/layers/{dataset_name}/"
    for partitioning in ["kmeans"]:
        # for partitioning in ["random", "louvain", "kmeans"]:
        file_name = f"20240520_203134_{dataset_name}"
        file_name_ = f"20240520_193025_{dataset_name}"
        # file_name_S = "final_result_S"
        file_name_S = f"{file_name}_S"
        file_name_S_ = f"{file_name_}_S"
        df_t, final_path_all = get_DF(path, partitioning, file_name, file_name_)
        df_S, final_path = get_DF(path, partitioning, file_name_S, file_name_S_)

        plt.plot(df_t.T, marker="*", label=df_t.index)
        plt.plot(df_S.T, marker="o", linestyle="--", label=df_t.index)
        plt.ylim([55, 85])
        plt.legend()
        df = pd.concat([df_t, df_S], axis=0)
        # df.T.plot()
        # df.T.plot(marker=["o", "o", "o", "o", "*", "*", "*", "*"])
        df.T.to_csv(final_path_all)
    plt.show()
