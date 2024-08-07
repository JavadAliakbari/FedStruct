import torch

from src import *
from src.GNN.sGNN import SClassifier
from src.utils.graph import Graph

# from src.GNN.MC import estimate


# class SLaplace(SClassifier):
#     def __init__(
#         self,
#         graph: Graph,
#         hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
#     ):
#         super().__init__(graph, hidden_layer_size)
#         self.create_L()
#         self.old_x = deepcopy(self.graph.x.detach())
#         self.initial_x = deepcopy(self.graph.x.detach())

#     def regularizer(self):
#         s = 0
#         x = self.graph.x
#         delta_x = x - self.old_x
#         # delta_x = x - self.initial_x

#         delta_x_i = delta_x[self.directed_edges[0]]
#         delta_x_o = delta_x[self.directed_edges[1]]
#         old_x_i = self.old_x[self.directed_edges[0]]
#         old_x_o = self.old_x[self.directed_edges[1]]
#         initial_x_i = self.initial_x[self.directed_edges[0]]
#         initial_x_o = self.initial_x[self.directed_edges[1]]
#         x_i = x[self.directed_edges[0]]
#         x_o = x[self.directed_edges[1]]
#         s += (
#             torch.sum(
#                 # (old_x_i - old_x_o) ** 2
#                 # + (delta_x_i - delta_x_o) ** 2
#                 +(x_i - x_o)
#                 * (delta_x_i - 0)
#                 # +(initial_x_i - initial_x_o)
#                 # * (delta_x_i - 0)
#                 # -(old_x_i - old_x_o)
#                 # * (delta_x_o - 0)
#             )
#             / x.shape[1]
#         )
#         # s += (
#         #     torch.sum(
#         #         # (old_x_i - old_x_o) ** 2
#         #         # + (delta_x_i - delta_x_o) ** 2
#         #         +(x_i - x_o)
#         #         # +(old_x_i - old_x_o)
#         #         * (delta_x_i - 0)
#         #     )
#         #     / x.shape[1]
#         # )
#         # s += torch.sum((old_x_i + delta_x_i - old_x_o - delta_x_o) ** 2) / x.shape[1]
#         # s += (
#         #     torch.sum(
#         #         (initial_x_i - initial_x_o) ** 2
#         #         + (delta_x_i - delta_x_o) ** 2
#         #         + 2 * (initial_x_i - 0) * (delta_x_i - delta_x_o)
#         #     )
#         #     / x.shape[1]
#         # )
#         # s += (
#         #     torch.sum((initial_x_i + delta_x_i - initial_x_o - delta_x_o) ** 2)
#         #     / x.shape[1]
#         # )

#         # initial_x_i = self.initial_x[self.directed_edges[0]]
#         # initial_x_o = self.initial_x[self.directed_edges[1]]
#         # x_i = x[self.directed_edges[0]]
#         # delta_x_i = delta_x[self.directed_edges[0]]
#         # s += torch.sum((initial_x_i - initial_x_o) * (x_i - delta_x_i)) / x.shape[1]

#         self.old_x = deepcopy(self.graph.x.detach())

#         # r1 = torch.matmul(self.L, x)
#         # r = torch.matmul(x.T, r1)
#         # s = torch.trace(r) / torch.trace(torch.matmul(x.T, x))
#         # s = torch.trace(r) / x.shape[1]

#         return s


class SLaplace(SClassifier):
    def __init__(
        self,
        graph: Graph,
        hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
    ):
        super().__init__(graph, hidden_layer_size)
        self.graph.create_L()

    def regularizer(self):
        x = self.graph.x
        r1 = torch.matmul(self.graph.L, x)
        r = torch.matmul(x.T, r1)
        s = torch.trace(r) / torch.trace(torch.matmul(x.T, x))
        # s = torch.trace(r) / x.shape[1]

        return s


class SpectralLaplace(SClassifier):
    def __init__(
        self,
        graph: Graph,
        hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
    ):
        super().__init__(graph, hidden_layer_size)

    def get_SFV(self):
        Q = self.calc_Qh()
        return torch.matmul(self.U, Q)

    def set_UD(self, U, D):
        self.U = U
        self.D = D

    def calc_Qh(self):
        return self.graph.x

    def regularizer(self):
        Q = self.calc_Qh()
        if len(self.D.shape) == 1:
            r1 = torch.einsum("i,ij->ij", self.D, Q)
            # r1 = torch.einsum(
            #     "i,ij->ij", (self.D) ** (config.structure_model.DGCN_layers), Q
            # )
        else:
            r1 = torch.einsum("ij,jk->ik", self.D, Q)
        r = torch.matmul(Q.T, r1)
        s = torch.trace(r) / torch.trace(torch.matmul(Q.T, Q))
        # s = torch.trace(r) / x.shape[1]

        return s

    def get_embeddings(self):
        Q = self.calc_Qh()
        SFV = torch.matmul(self.U, Q)
        torch.nn.functional.relu(SFV, inplace=True)
        H = self.model(SFV)
        return H


class LanczosLaplace(SpectralLaplace):
    def __init__(
        self,
        graph: Graph,
        hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
    ):
        super().__init__(graph, hidden_layer_size)

    def set_AV(self, V, A):
        self.V = V
        self.A = A

    def guess(self, y):
        mask = torch.arange(y.shape[0]).unsqueeze(1).eq(self.graph.node_ids).any(1)
        Q = self.calc_Qh()
        SFV = torch.matmul(self.V, Q)
        H = self.model(SFV)
        y_pred = torch.nn.functional.softmax(H, dim=1)
        acc = calc_accuracy(y_pred.argmax(dim=1)[mask], y[mask])
        SFV2 = torch.matmul(self.A, Q)
        H2 = self.model(SFV2)
        y_pred2 = torch.nn.functional.softmax(H2, dim=1)
        acc2 = calc_accuracy(y_pred2.argmax(dim=1)[mask], y[mask])
        return acc, acc2


# class SLaplace(SClassifier):
#     def __init__(
#         self,
#         graph: Graph,
#         hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
#     ):
#         super().__init__(graph, hidden_layer_size)
#         A, E = estimate(self.graph, self.graph.x.shape[0])
#         self.A = A + E
#         D = torch.sum(self.A, dim=1)
#         self.L = torch.diag(D) - self.A
#         # self.create_L()

#     def regularizer(self):
#         x = self.graph.x
#         r1 = torch.matmul(self.L, x)
#         r = torch.matmul(x.T, r1)
#         # s = torch.trace(r) / torch.trace(torch.matmul(x.T, x))
#         s = torch.trace(r) / x.shape[1]

#         return s

#     def get_grads(self, just_SFV=False):
#         grads = {}
#         if self.model is not None:
#             grads["model"] = self.model.get_grads()

#         return grads

#     def set_grads(self, grads):
#         if "model" in grads.keys():
#             self.model.set_grads(grads["model"])
