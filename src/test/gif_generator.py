import os

import imageio


def generate_gif(root_path):
    folder_path = f"{root_path}TSNE/"
    filenames = os.listdir(folder_path)
    filenames = [filename for filename in filenames if filename.endswith(".png")]
    filenames = list(sorted(filenames, key=lambda elem: int(elem.split(".")[0][6:])))

    images = []
    for filename in filenames:
        file_path = f"{folder_path}{filename}"
        images.append(imageio.imread(file_path))
    gif_path = f"{root_path}/gif/"
    os.makedirs(gif_path, exist_ok=True)
    imageio.mimsave(f"{gif_path}/movie.gif", images, format="GIF", fps=5)


base_path = f"results/"
dataset = "Cora"
structure_type = "hop2vec"
partioning = "random"
# smodel = "MLP"
smodel = "Laplace"
now = "20240616_154202"
# now = "20240612_131744"
# now = "20240612_130054"
num_subgraphs = 10
root_path = f"{base_path}{dataset}/{structure_type}/{partioning}/{smodel}/{num_subgraphs}/all/plots/{now}/"
generate_gif(root_path)
