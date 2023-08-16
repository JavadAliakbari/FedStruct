import os
from copy import deepcopy

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [24, 16]
plt.rcParams["figure.dpi"] = 100  # 200 e.g. is really fine, but slower
plt.rcParams.update({"figure.max_open_warning": 0})
plt.rcParams["font.size"] = 20


def add_weights(sum_weights, weights):
    if sum_weights is None:
        sum_weights = deepcopy(weights)
    else:
        for layer_name, layer_parameters in weights.items():
            for component_name, component_parameters in layer_parameters.items():
                sum_weights[layer_name][component_name] += component_parameters
    return sum_weights


def calc_mean_weights(sum_weights, count):
    for layer_name, layer_parameters in sum_weights.items():
        for component_name, component_parameters in layer_parameters.items():
            sum_weights[layer_name][component_name] = component_parameters / count

    return sum_weights


def plot_metrics(
    res,
    title="",
    save_path="./",
):
    dataset = pd.DataFrame.from_dict(res)
    dataset.set_index("Epoch", inplace=True)

    # save_dir = f"./plot_results/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    loss_columns = list(filter(lambda x: x.endswith("Loss"), dataset.columns))
    dataset[loss_columns].plot()
    plot_title = f"loss {title}"
    plt.title(plot_title)
    plt.savefig(f"{save_path}{plot_title}.png")

    acc_columns = list(filter(lambda x: x.endswith("Acc"), dataset.columns))
    dataset[acc_columns].plot()
    plot_title = f"accuracy {title}"
    plt.title(plot_title)
    plt.savefig(f"{save_path}{plot_title}.png")
