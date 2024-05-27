#!/bin/bash
if [ ! -d "venv16" ]; then
    PWD=`pwd`
    python -m venv venv16
    echo "before calling source: $PATH"
    echo $PWD
    activate () {
        . $PWD/venv16/bin/activate
    }
    activate
    echo "after calling source: $PATH"
    $PWD/venv16/bin/pip install --upgrade pip
    $PWD/venv16/bin/pip3 install --no-cache-dir torch torchvision torchaudio
    $PWD/venv16/bin/python -m pip --no-cache-dir  install  torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html 
    $PWD/venv16/bin/python -m pip --no-cache-dir  install  torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
    $PWD/venv16/bin/python -m pip --no-cache-dir  install  torch_cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
    # python -m pip --no-cache-dir  install  torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
    $PWD/venv16/bin/python -m pip --no-cache-dir  install torch_geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
    $PWD/venv16/bin/python -m pip install pandas
    $PWD/venv16/bin/python -m pip install matplotlib
    $PWD/venv16/bin/python -m pip install pyyaml
fi

CONFIG_PATH="./config/config_Cora.yml" python main.py