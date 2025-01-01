import os
import sys
import json

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)


from src import *
from src.GNN.GNN_server import GNNServer
from src.utils.define_graph import define_graph
from src.utils.graph_partitioning import partition_graph


def set_up_system():
    LOGGER.info(json.dumps(config.config, indent=4))

    graph = define_graph(config.dataset.dataset_name)

    graph.calc_abar(
        config.structure_model.DGCN_layers,
        pruning=config.subgraph.prune,
    )

    graph.add_masks(
        train_ratio=config.subgraph.train_ratio,
        test_ratio=config.subgraph.test_ratio,
    )

    subgraphs = partition_graph(
        graph, config.subgraph.num_subgraphs, config.subgraph.partitioning
    )

    GNN_server = GNNServer(graph)
    for subgraph in subgraphs:
        GNN_server.add_client(subgraph)

    results = {}

    normal_epoch = config.model.iterations
    laplace_epoch = 200
    spectral_epoch1 = laplace_epoch
    spectral_epoch2 = 45

    res = GNN_server.joint_train_g(
        epochs=normal_epoch,
        data_type="feature",
        smodel_type="GNN",
        FL=True,
    )
    results[f"FL S MLP"] = round(res["Average"]["Test Acc"], 4)

    res = GNN_server.joint_train_g(
        epochs=normal_epoch,
        data_type="structure",
        smodel_type="MLP",
        FL=True,
    )
    results[f"FL S MLP"] = round(res["Average"]["Test Acc"], 4)

    res = GNN_server.joint_train_g(
        epochs=normal_epoch,
        data_type="structure",
        smodel_type="DGCN",
        FL=True,
    )
    results[f"FL S DGCN"] = round(res["Average"]["Test Acc"], 4)

    res = GNN_server.joint_train_g(
        epochs=laplace_epoch,
        data_type="structure",
        smodel_type="Laplace",
        FL=True,
    )
    results[f"FL S Laplacian"] = round(res["Average"]["Test Acc"], 4)

    res = GNN_server.joint_train_g(
        epochs=spectral_epoch2,
        data_type="structure",
        smodel_type="SpectralLaplace",
        FL=True,
    )
    results[f"FL S Spectral"] = round(res["Average"]["Test Acc"], 4)

    res = GNN_server.joint_train_g(
        epochs=spectral_epoch2,
        data_type="structure",
        smodel_type="LanczosLaplace",
        FL=True,
    )
    results[f"FL S Lanczos"] = round(res["Average"]["Test Acc"], 4)
    ####################################################################

    res = GNN_server.joint_train_g(
        epochs=normal_epoch,
        data_type="f+s",
        smodel_type="MLP",
        fmodel_type="DGCN",
        FL=True,
    )
    results[f"FL F+S MLP"] = round(res["Average"]["Test Acc"], 4)
    results[f"FL F+S(F) MLP"] = round(res["Average"]["Test Acc F"], 4)
    results[f"FL F+S(S) MLP"] = round(res["Average"]["Test Acc S"], 4)

    res = GNN_server.joint_train_g(
        epochs=normal_epoch,
        data_type="f+s",
        smodel_type="DGCN",
        fmodel_type="DGCN",
        FL=True,
    )
    results[f"FL F+S DGCN"] = round(res["Average"]["Test Acc"], 4)
    results[f"FL F+S(F) DGCN"] = round(res["Average"]["Test Acc F"], 4)
    results[f"FL F+S(S) DGCN"] = round(res["Average"]["Test Acc S"], 4)

    res = GNN_server.joint_train_g(
        epochs=laplace_epoch,
        data_type="f+s",
        smodel_type="Laplace",
        fmodel_type="DGCN",
        FL=True,
    )
    results[f"FL F+S Laplacian"] = round(res["Average"]["Test Acc"], 4)
    results[f"FL F+S(F) Laplacian"] = round(res["Average"]["Test Acc F"], 4)
    results[f"FL F+S(S) Laplacian"] = round(res["Average"]["Test Acc S"], 4)

    res = GNN_server.joint_train_g(
        epochs=normal_epoch,
        data_type="f+s",
        smodel_type="SpectralLaplace",
        fmodel_type="DGCN",
        FL=True,
        spectral_len=0,
    )
    results[f"FL F+S SpectralLaplace(C)"] = round(res["Average"]["Test Acc"], 4)
    results[f"FL F+S(F) SpectralLaplace(C)"] = round(res["Average"]["Test Acc F"], 4)
    results[f"FL F+S(S) SpectralLaplace(C)"] = round(res["Average"]["Test Acc S"], 4)

    res = GNN_server.joint_train_g(
        epochs=spectral_epoch2,
        data_type="f+s",
        smodel_type="SpectralLaplace",
        fmodel_type="DGCN",
        FL=True,
    )

    results[f"FL F+S SpectralLaplace"] = round(res["Average"]["Test Acc"], 4)
    results[f"FL F+S(F) SpectralLaplace"] = round(res["Average"]["Test Acc F"], 4)
    results[f"FL F+S(S) SpectralLaplace"] = round(res["Average"]["Test Acc S"], 4)

    res = GNN_server.joint_train_g(
        epochs=spectral_epoch2,
        data_type="f+s",
        smodel_type="LanczosLaplace",
        fmodel_type="DGCN",
        FL=True,
    )
    results[f"FL F+S LanczosLaplace"] = round(res["Average"]["Test Acc"], 4)
    results[f"FL F+S(F) LanczosLaplace"] = round(res["Average"]["Test Acc F"], 4)
    results[f"FL F+S(S) LanczosLaplace"] = round(res["Average"]["Test Acc S"], 4)

    res = GNN_server.joint_train_g(
        epochs=spectral_epoch2,
        data_type="f+s",
        smodel_type="SpectralDGCN",
        fmodel_type="DGCN",
        FL=True,
    )
    results[f"FL F+S SpectralDGCN"] = round(res["Average"]["Test Acc"], 4)
    results[f"FL F+S(F) SpectralDGCN"] = round(res["Average"]["Test Acc F"], 4)
    results[f"FL F+S(S) SpectralDGCN"] = round(res["Average"]["Test Acc S"], 4)

    res = GNN_server.joint_train_g(
        epochs=spectral_epoch2,
        data_type="f+s",
        smodel_type="LanczosDGCN",
        fmodel_type="DGCN",
        FL=True,
    )
    results[f"FL F+S LanczosDGCN"] = round(res["Average"]["Test Acc"], 4)
    results[f"FL F+S(F) LanczosDGCN"] = round(res["Average"]["Test Acc F"], 4)
    results[f"FL F+S(S) LanczosDGCN"] = round(res["Average"]["Test Acc S"], 4)

    LOGGER.info(json.dumps(results, indent=4))


if __name__ == "__main__":
    set_up_system()
    # plt.show()
