if [ ! -d ".venv3" ]; then
#   echo "$DIRECTORY does not exist."
    python -m venv .venv3
    source .venv/bin/activate
    pip install --upgrade pip
    pip3 --no-cache-dir install torch torchvision torchaudio
    python -m pip --no-cache-dir  install  torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html 
    python -m pip --no-cache-dir  install  torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
    python -m pip --no-cache-dir  install  torch_cluster -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
    python -m pip --no-cache-dir  install  torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
    python -m pip --no-cache-dir  install torch_geometric -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
    pip install pandas
    pip install matplotlib
    pip install pyyaml
fi

CONFIG_PATH="./config/config_Cora.yml" python main.py