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

    graph.obtain_a(
        config.structure_model.DGCN_layers,
        estimate=config.structure_model.estimate,
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

    normal_epoch = 150
    laplace_epoch=250
    spectral_epoch1= laplace_epoch
    spectral_epoch2 = 75

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
        epochs=spectral_epoch2,
        data_type="f+s",
        smodel_type="SpectralLaplace",
        fmodel_type="DGCN",
        FL=True,
    )
    results[f"FL F+S Spectral"] = round(res["Average"]["Test Acc"], 4)
    results[f"FL F+S(F) Spectral"] = round(res["Average"]["Test Acc F"], 4)
    results[f"FL F+S(S) Spectral"] = round(res["Average"]["Test Acc S"], 4)

    res = GNN_server.joint_train_g(
        epochs=spectral_epoch2,
        data_type="f+s",
        smodel_type="LanczosLaplace",
        fmodel_type="DGCN",
        FL=True,
    )
    results[f"FL F+S Lanczos"] = round(res["Average"]["Test Acc"], 4)
    results[f"FL F+S(F) Lanczos"] = round(res["Average"]["Test Acc F"], 4)
    results[f"FL F+S(S) Lanczos"] = round(res["Average"]["Test Acc S"], 4)

    LOGGER.info(json.dumps(results, indent=4))


if __name__ == "__main__":
    set_up_system()
    # plt.show()