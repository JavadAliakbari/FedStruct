import pandas as pd

if __name__ == "__main__":
    # folder_path = "results/Paper Results/Cora/louvain-10/0.1/"
    for dataset in [
        "Cora",
        "CiteSeer",
        "PubMed",
        "Chameleon",
        "Photo",
        "Amazon-ratings",
    ]:
        for partioning in ["louvain", "kmeans", "hop2vec"]:
            if partioning == "hop2vec":
                num_subgraphs_list = [5, 10, 20]
            else:
                num_subgraphs_list = [10]
            # num_subgraphs_list = [5,10,20]
            for num_subgraphs in num_subgraphs_list:
                folder_path = f"ICML Results/Paper Results48/{dataset}/{partioning}/{num_subgraphs}/0.1/"
                # folder_path = f"results/Paper Results2/{dataset}/{partioning}/{num_subgraphs}/0.1/"
                path = f"{folder_path}final_result_0.1.csv"
                df = pd.read_csv(path, index_col="Unnamed: 0")
                # rows = [
                #     # "server_mlp",
                #     "flga_mlp",
                #     "flga_GNN",
                #     "fedsage+_GA_GNN",
                #     "fedsage_0_GNN",
                #     "fedsage_ideal_GNN",
                #     "degree_sdga_DGCN",
                #     "GDV_sdga_DGCN",
                #     "node2vec_sdga_DGCN",
                #     "hop2vec_sdga_DGCN",
                #     "degree_sdga_GNN",
                #     "GDV_sdga_GNN",
                #     "node2vec_sdga_GNN",
                #     "hop2vec_sdga_GNN",
                #     "local_GNN",
                #     "local_mlp",
                #     # "server_GNN",
                # ]
                rows = [
                    # "server_mlp",
                    # "flga_mlp",
                    "flga_GNN",
                    "flga_DGCN",
                    "fedsage+_GA_GNN",
                    # "fedsage_0_GNN",
                    "fedsage_ideal_GNN",
                    "degree_sdga_DGCN",
                    "GDV_sdga_DGCN",
                    "node2vec_sdga_DGCN",
                    "hop2vec_sdga_DGCN",
                    # "degree_sdga_GNN",
                    # "GDV_sdga_GNN",
                    # "node2vec_sdga_GNN",
                    # "hop2vec_sdga_GNN",
                    "local_GNN",
                    "local_DGCN",
                    # "local_mlp",
                    # "server_GNN",
                ]
                # rows = [
                #     "flga_degree_DGCN_true",
                #     "flga_GDV_DGCN_true",
                #     "flga_node2vec_DGCN_true",
                #     "flga_hop2vec_DGCN_true",
                #     "flga_degree_DGCN_prune",
                #     "flga_GDV_DGCN_prune",
                #     "flga_node2vec_DGCN_prune",
                #     "flga_hop2vec_DGCN_prune",
                # ]
                df2 = df.loc[rows, "0"].tolist()
                data = [x.split("±") for x in df2]
                data = [[100 * float(x[0]), 100 * float(x[1])] for x in data]
                data = [rf"{x[0]:0.2f}$\pm$ {x[1]:0.2f}" for x in data]
                with open(f"{folder_path}paper_DGCN.txt", "w") as fid:
                    for line in data:
                        fid.write(f"{line}\n")

        a = 1
