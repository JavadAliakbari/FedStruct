from src.FedPub.utils import *
from torch_geometric.loader import DataLoader

from src.utils.graph import Graph


class PubDataLoader:
    def __init__(self, graph: Graph):
        self.pa_loader = [graph]
        # self.pa_loader = DataLoader(
        #     dataset=[graph],
        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=0,
        #     pin_memory=False,
        # )
