#!/bin/bash
if [ ! -d ".venv" ]; then
#   echo "$DIRECTORY does not exist."
    conda create -n GNN_M1  python=3.10
    echo "before calling source: $PATH"
    eval "$(conda shell.bash hook)"
    conda activate GNN_M1
    echo "after calling source: $PATH"
    pip install --upgrade pip
    pip3 install --no-cache-dir torch torchvision torchaudio
    python -m pip --no-cache-dir  install  torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html 
    python -m pip --no-cache-dir  install  torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
    python -m pip --no-cache-dir  install  torch_cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
    # python -m pip --no-cache-dir  install  torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
    python -m pip --no-cache-dir  install torch_geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
    python -m pip install pandas
    python -m pip install matplotlib
    python -m pip install pyyaml
fi

CONFIG_PATH="./config/config_Cora.yml" python main.py