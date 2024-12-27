import os
import sys

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

from src import *
from src.utils.define_graph import define_graph
from matplotlib.colors import LinearSegmentedColormap


if __name__ == "__main__":
    # Define custom colors for 0 and 1 (e.g., 0 -> blue, 1 -> red)
    colors = [
        (0, "white"),
        (1, "blue"),
    ]  # Replace 'blue' and 'red' with your desired colors

    # Create a custom colormap using the colors
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    print(plt.colormaps())
    graph = define_graph("Cora")
    graph.add_masks(
        train_ratio=config.subgraph.train_ratio,
        test_ratio=config.subgraph.test_ratio,
    )
    abar = graph.calc_abar()
    graph.DGCN_abar = None
    abar_prune = graph.calc_abar(pruning=True)

    # plot_abar2(abar, graph.y, name="Photo_orig", save=True)
    # plot_abar2(abar_prune, graph.y, name="Photo_prunnnne", save=True)
    plot_abar(abar, graph.edge_index, name="Cora_orig", save=True, cmap=custom_cmap)
    plot_abar(
        abar_prune,
        graph.edge_index,
        name="Cora_prunnnne",
        save=True,
        cmap=custom_cmap,
    )
    # fig.imsave(f"./models/graph.png", dense_abar)
    # plot_abar2(abar, graph.y.numpy())
