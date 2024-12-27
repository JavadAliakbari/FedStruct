import torch
import numpy as np


def find_neighbors_(
    node_id: int,
    edge_index,
    include_node=False,
):
    all_neighbors = np.unique(
        np.hstack(
            (
                edge_index[1, edge_index[0] == node_id],
                edge_index[0, edge_index[1] == node_id],
            )
        )
    )

    if not include_node:
        all_neighbors = np.setdiff1d(all_neighbors, node_id)

    return all_neighbors


class Pair:
    def __init__(self, a, b, n):
        a0 = min(a, b)
        b0 = max(b, a)
        self.key = a0 * n + b0


class Triple:
    def __init__(self, a, b, c, n):
        a0 = a
        b0 = b
        c0 = c

        if a0 > b0:
            a0, b0 = Triple.swap(a0, b0)
        if b0 > c0:
            b0, c0 = Triple.swap(b0, c0)
        if a0 > b0:
            a0, b0 = Triple.swap(a0, b0)

        self.key = a0 * n * n + b0 * n + c0

    def swap(a, b):
        temp = a
        a = b
        b = temp

        return a, b


class GDV:
    def __init__(
        self,
    ):
        pass

    def adjacent(self, a, b):
        if b in self.set_adj[a]:
            return True
        return False

    def make_directed(edges):
        new_edges = edges.T.tolist()
        i = 0
        while i < len(new_edges):
            a, b = new_edges[i]
            new_edges = list(filter(lambda x: not (x[0] == b and x[1] == a), new_edges))
            i += 1

        return torch.tensor(np.array(new_edges, dtype=int).T)

    def count5(
        self,
        edges,
        n=None,
    ):
        edges = edges.clone().cpu()
        if n is None:
            n = max(edges.flatten()).item() + 1
        # edges = GDV.make_directed(edges)
        m = edges.shape[1]

        self.adj = []
        self.set_adj = []
        self.deg = []
        for i in range(n):
            neighbors = find_neighbors_(i, edges).tolist()
            self.adj.append(neighbors)
            self.set_adj.append(set(neighbors))
            self.deg.append(len(neighbors))

        self.degree_sorted_adj = [
            sorted(elem, key=lambda x: self.deg[x]) for elem in self.adj
        ]

        inc = []
        for i in range(n):
            inc.append([])

        for i in range(m):
            a, b = edges[:, i].tolist()
            inc[a].append([b, i])
            # inc[b].append([a, i])

        inc = [sorted(elem, key=lambda x: self.deg[x[0]]) for elem in inc]

        common2 = dict()
        common3 = dict()
        orbit = np.zeros((n, 73), dtype=int)

        for x in range(n):
            for n1 in range(self.deg[x]):
                a = self.degree_sorted_adj[x][n1]
                for n2 in range(n1 + 1, self.deg[x]):
                    b = self.degree_sorted_adj[x][n2]
                    if not self.adjacent(a, b):
                        continue
                    # ab = a * n + b
                    ab = Pair(a, b, n).key
                    if ab not in common2.keys():
                        common2[ab] = 1
                    else:
                        common2[ab] += 1

                    for n3 in range(n2 + 1, self.deg[x]):
                        c = self.degree_sorted_adj[x][n3]
                        if not self.adjacent(a, c):
                            continue
                        if not self.adjacent(b, c):
                            continue

                        # abc = a * n * n + b * n + c
                        abc = Triple(a, b, c, n).key
                        if abc not in common3.keys():
                            common3[abc] = 1
                        else:
                            common3[abc] += 1

        tri = m * [0]
        for i in range(m):
            x, y = edges[:, i].tolist()

            common_neighbors = self.set_adj[x].intersection(self.set_adj[y])
            tri[i] = len(common_neighbors)

        C5 = n * [0]
        neigh = n * [0]
        neigh2 = n * [0]

        for x in range(n):
            for nx in range(self.deg[x]):
                y = self.adj[x][nx]
                if y >= x:
                    break

                nn = 0
                for ny in range(self.deg[y]):
                    z = self.adj[y][ny]
                    if z >= y:
                        break
                    if self.adjacent(x, z):
                        neigh[nn] = z
                        nn += 1

                for i in range(nn):
                    z = neigh[i]
                    nn2 = 0
                    for j in range(i + 1, nn):
                        zz = neigh[j]
                        if self.adjacent(z, zz):
                            neigh2[nn2] = zz
                            nn2 += 1

                    for i2 in range(nn2):
                        zz = neigh2[i2]
                        for j2 in range(i2 + 1, nn2):
                            zzz = neigh2[j2]
                            if self.adjacent(zz, zzz):
                                C5[x] += 1
                                C5[y] += 1
                                C5[z] += 1
                                C5[zz] += 1
                                C5[zzz] += 1

        common_x = n * [0]
        common_x_list = n * [0]
        common_a = n * [0]
        common_a_list = n * [0]
        ncx = 0
        nca = 0

        neigh_x = np.zeros(n, dtype=bool)
        neigh_a = np.zeros(n, dtype=bool)

        for x in range(n):
            for i in range(ncx):
                common_x[common_x_list[i]] = 0

            ncx = 0

            neigh_x[self.degree_sorted_adj[x]] = True

            # smaller graphlets
            orbit[x][0] = self.deg[x]
            for nx1 in range(self.deg[x]):
                a = self.degree_sorted_adj[x][nx1]
                for nx2 in range(nx1 + 1, self.deg[x]):
                    b = self.degree_sorted_adj[x][nx2]
                    if self.adjacent(a, b):
                        orbit[x][3] += 1
                    else:
                        orbit[x][2] += 1

                for na in range(self.deg[a]):
                    b = self.degree_sorted_adj[a][na]
                    if b != x and not neigh_x[b]:
                        orbit[x][1] += 1
                        if common_x[b] == 0:
                            common_x_list[ncx] = b
                            ncx += 1

                        common_x[b] += 1

            f = 72 * [0]

            for nx1 in range(self.deg[x]):
                a, xa = inc[x][nx1]

                neigh_a[self.degree_sorted_adj[a]] = True

                for i in range(nca):
                    common_a[common_a_list[i]] = 0
                nca = 0

                for na in range(self.deg[a]):
                    b = self.degree_sorted_adj[a][na]
                    for nb in range(self.deg[b]):
                        c = self.degree_sorted_adj[b][nb]
                        if c == a or neigh_a[c]:
                            continue
                        if common_a[c] == 0:
                            common_a_list[nca] = c
                            nca += 1

                        common_a[c] += 1

                # x = orbit-14 (tetrahedron)
                for nx2 in range(nx1 + 1, self.deg[x]):
                    b, xb = inc[x][nx2]

                    if not neigh_a[b]:
                        continue
                    for nx3 in range(nx2 + 1, self.deg[x]):
                        c, xc = inc[x][nx3]
                        if not neigh_a[c] or not self.adjacent(b, c):
                            continue
                        orbit[x][14] += 1
                        f[70] += common3.get(Triple(a, b, c, n).key, 0) - 1
                        if tri[xa] > 2 and tri[xb] > 2:
                            f[71] += common3.get(Triple(x, a, b, n).key, 0) - 1
                        if tri[xa] > 2 and tri[xc] > 2:
                            f[71] += common3.get(Triple(x, a, c, n).key, 0) - 1
                        if tri[xb] > 2 and tri[xc] > 2:
                            f[71] += common3.get(Triple(x, b, c, n).key, 0) - 1
                        f[67] += tri[xa] - 2 + tri[xb] - 2 + tri[xc] - 2
                        f[66] += common2.get(Pair(a, b, n).key, 0) - 2
                        f[66] += common2.get(Pair(a, c, n).key, 0) - 2
                        f[66] += common2.get(Pair(b, c, n).key, 0) - 2
                        f[58] += self.deg[x] - 3
                        f[57] += self.deg[a] - 3 + self.deg[b] - 3 + self.deg[c] - 3

                # x = orbit-13 (diamond)
                for nx2 in range(0, self.deg[x]):
                    b, xb = inc[x][nx2]

                    if not neigh_a[b]:
                        continue
                    for nx3 in range(nx2 + 1, self.deg[x]):
                        c, xc = inc[x][nx3]
                        if not neigh_a[c] or self.adjacent(b, c):
                            continue
                        orbit[x][13] += 1
                        bc_is = self.set_adj[b].intersection(self.set_adj[c])
                        xbc_is = bc_is.intersection(self.set_adj[x])
                        abc_is = bc_is.intersection(self.set_adj[a])
                        if tri[xb] > 1 and tri[xc] > 1:
                            f[69] += len(xbc_is) - 1
                        # f[68] += (
                        #     common3.get(Triple(a, b, c, n).key, 0) - 1
                        # )  # exception
                        f[68] += len(abc_is) - 1
                        # f[64] += common2.get(Pair(b, c, n).key, 0) - 2  # exception
                        f[64] += len(bc_is) - 2
                        f[61] += tri[xb] - 1 + tri[xc] - 1
                        f[60] += common2.get(Pair(a, b, n).key, 0) - 1
                        f[60] += common2.get(Pair(a, c, n).key, 0) - 1
                        f[55] += tri[xa] - 2
                        f[48] += self.deg[b] - 2 + self.deg[c] - 2
                        f[42] += self.deg[x] - 3
                        f[41] += self.deg[a] - 3

                # x = orbit-12 (diamond)
                for nx2 in range(nx1 + 1, self.deg[x]):
                    b, xb = inc[x][nx2]

                    if not neigh_a[b]:
                        continue
                    for na in range(self.deg[a]):
                        c, ac = inc[a][na]
                        if c == x or neigh_x[c] or not self.adjacent(b, c):
                            continue
                        orbit[x][12] += 1
                        if tri[ac] > 1:
                            f[65] += common3.get(Triple(a, b, c, n).key, 0)
                        f[63] += common_x[c] - 2
                        f[59] += tri[ac] - 1 + common2.get(Pair(b, c, n).key, 0) - 1
                        f[54] += common2.get(Pair(a, b, n).key, 0) - 2
                        f[47] += self.deg[x] - 2
                        f[46] += self.deg[c] - 2
                        f[40] += self.deg[a] - 3 + self.deg[b] - 3

                # x = orbit-8 (cycle)
                for nx2 in range(nx1 + 1, self.deg[x]):
                    b, xb = inc[x][nx2]

                    if neigh_a[b]:
                        continue
                    for na in range(self.deg[a]):
                        c, ac = inc[a][na]
                        if c == x or neigh_x[c] or not self.adjacent(b, c):
                            continue
                        orbit[x][8] += 1
                        if tri[ac] > 0:
                            # f[62] += common3.get(Triple(a, b, c, n).key, 0)  # exception
                            f[62] += len(
                                self.set_adj[a].intersection(
                                    self.set_adj[b], self.set_adj[c]
                                )
                            )
                        f[53] += tri[xa] + tri[xb]
                        f[51] += tri[ac] + common2.get(Pair(b, c, n).key, 0)
                        f[50] += common_x[c] - 2
                        f[49] += common_a[b] - 2
                        f[38] += self.deg[x] - 2
                        f[37] += self.deg[a] - 2 + self.deg[b] - 2
                        f[36] += self.deg[c] - 2

                # x = orbit-11 (paw)
                for nx2 in range(nx1 + 1, self.deg[x]):
                    b, xb = inc[x][nx2]

                    if not neigh_a[b]:
                        continue
                    for nx3 in range(0, self.deg[x]):
                        c, xc = inc[x][nx3]
                        if c == a or c == b or neigh_a[c] or self.adjacent(b, c):
                            continue
                        orbit[x][11] += 1
                        f[44] += tri[xc]
                        f[33] += self.deg[x] - 3
                        f[30] += self.deg[c] - 1
                        f[26] += self.deg[a] - 2 + self.deg[b] - 2

                # x = orbit-10 (paw)
                for nx2 in range(0, self.deg[x]):
                    b, xb = inc[x][nx2]

                    if not neigh_a[b]:
                        continue
                    for nb in range(0, self.deg[b]):
                        c, bc = inc[b][nb]
                        if c == x or c == a or neigh_a[c] or neigh_x[c]:
                            continue
                        orbit[x][10] += 1
                        f[52] += common_a[c] - 1
                        f[43] += tri[bc]
                        f[32] += self.deg[b] - 3
                        f[29] += self.deg[c] - 1
                        f[25] += self.deg[a] - 2

                # x = orbit-9 (paw) orbit-6 (claw)
                for na1 in range(0, self.deg[a]):
                    b, ab = inc[a][na1]

                    if b == x or neigh_x[b]:
                        continue
                    for na2 in range(na1 + 1, self.deg[a]):
                        c, ac = inc[a][na2]
                        if c == x or neigh_x[c]:
                            continue

                        da = self.deg[a] - 3
                        dx = self.deg[x] - 1
                        dbc = self.deg[b] - 1 + self.deg[c] - 1

                        if self.adjacent(b, c):
                            orbit[x][9] += 1
                            if tri[ab] > 1 and tri[ac] > 1:
                                f[56] += common3.get(Triple(a, b, c, n).key, 0)
                            bc_pair = Pair(b, c, n).key
                            f[45] += common2.get(bc_pair, 0) - 1
                            f[39] += tri[ab] - 1 + tri[ac] - 1
                            f[31] += da
                            f[28] += dx
                            f[24] += dbc - 2
                        else:
                            orbit[x][6] += 1
                            f[22] += da
                            f[20] += dx
                            f[19] += dbc

                # x = orbit-4 (path)
                for na in range(self.deg[a]):
                    b, ab = inc[a][na]

                    if b == x or neigh_x[b]:
                        continue
                    for nb in range(self.deg[b]):
                        c, bc = inc[b][nb]
                        if c == a or neigh_a[c] or neigh_x[c]:
                            continue
                        orbit[x][4] += 1
                        f[35] += common_a[c] - 1
                        f[34] += common_x[c]
                        f[27] += tri[bc]
                        f[18] += self.deg[b] - 2
                        f[16] += self.deg[x] - 1
                        f[15] += self.deg[c] - 1

                # x = orbit-5 (path)
                for nx2 in range(0, self.deg[x]):
                    b, xb = inc[x][nx2]

                    if b == a or neigh_a[b]:
                        continue
                    for nb in range(0, self.deg[b]):
                        c, bc = inc[b][nb]
                        if c == x or neigh_a[c] or neigh_x[c]:
                            continue
                        orbit[x][5] += 1
                        f[17] += self.deg[a] - 1

                # x = orbit-7 (claw)
                for nx2 in range(nx1 + 1, self.deg[x]):
                    b = self.degree_sorted_adj[x][nx2]

                    if neigh_a[b]:
                        continue
                    for nx3 in range(nx2 + 1, self.deg[x]):
                        c = self.degree_sorted_adj[x][nx3]
                        if neigh_a[c] or self.adjacent(b, c):
                            continue
                        orbit[x][7] += 1
                        f[23] += self.deg[x] - 3
                        f[21] += self.deg[a] - 1 + self.deg[b] - 1 + self.deg[c] - 1

                # reset neigh_a
                neigh_a[self.degree_sorted_adj[a]] = False

            # solve equations
            orbit[x][72] = C5[x]
            orbit[x][71] = (f[71] - 12 * orbit[x][72]) / 2
            orbit[x][70] = f[70] - 4 * orbit[x][72]
            orbit[x][69] = (f[69] - 2 * orbit[x][71]) / 4
            orbit[x][68] = f[68] - 2 * orbit[x][71]
            orbit[x][67] = f[67] - 12 * orbit[x][72] - 4 * orbit[x][71]
            orbit[x][66] = (
                f[66] - 12 * orbit[x][72] - 2 * orbit[x][71] - 3 * orbit[x][70]
            )
            orbit[x][65] = (f[65] - 3 * orbit[x][70]) / 2
            orbit[x][64] = (
                f[64] - 2 * orbit[x][71] - 4 * orbit[x][69] - 1 * orbit[x][68]
            )
            orbit[x][63] = f[63] - 3 * orbit[x][70] - 2 * orbit[x][68]
            orbit[x][62] = (f[62] - 1 * orbit[x][68]) / 2
            orbit[x][61] = (
                f[61] - 4 * orbit[x][71] - 8 * orbit[x][69] - 2 * orbit[x][67]
            ) / 2
            orbit[x][60] = (
                f[60] - 4 * orbit[x][71] - 2 * orbit[x][68] - 2 * orbit[x][67]
            )
            orbit[x][59] = (
                f[59] - 6 * orbit[x][70] - 2 * orbit[x][68] - 4 * orbit[x][65]
            )
            orbit[x][58] = (
                f[58] - 4 * orbit[x][72] - 2 * orbit[x][71] - 1 * orbit[x][67]
            )
            orbit[x][57] = (
                f[57]
                - 12 * orbit[x][72]
                - 4 * orbit[x][71]
                - 3 * orbit[x][70]
                - 1 * orbit[x][67]
                - 2 * orbit[x][66]
            )
            orbit[x][56] = (f[56] - 2 * orbit[x][65]) / 3
            orbit[x][55] = (f[55] - 2 * orbit[x][71] - 2 * orbit[x][67]) / 3
            orbit[x][54] = (
                f[54] - 3 * orbit[x][70] - 1 * orbit[x][66] - 2 * orbit[x][65]
            ) / 2
            orbit[x][53] = (
                f[53] - 2 * orbit[x][68] - 2 * orbit[x][64] - 2 * orbit[x][63]
            )
            orbit[x][52] = (
                f[52] - 2 * orbit[x][66] - 2 * orbit[x][64] - 1 * orbit[x][59]
            ) / 2
            orbit[x][51] = (
                f[51] - 2 * orbit[x][68] - 2 * orbit[x][63] - 4 * orbit[x][62]
            )
            orbit[x][50] = (f[50] - 1 * orbit[x][68] - 2 * orbit[x][63]) / 3
            orbit[x][49] = (
                f[49] - 1 * orbit[x][68] - 1 * orbit[x][64] - 2 * orbit[x][62]
            ) / 2
            orbit[x][48] = (
                f[48]
                - 4 * orbit[x][71]
                - 8 * orbit[x][69]
                - 2 * orbit[x][68]
                - 2 * orbit[x][67]
                - 2 * orbit[x][64]
                - 2 * orbit[x][61]
                - 1 * orbit[x][60]
            )
            orbit[x][47] = (
                f[47]
                - 3 * orbit[x][70]
                - 2 * orbit[x][68]
                - 1 * orbit[x][66]
                - 1 * orbit[x][63]
                - 1 * orbit[x][60]
            )
            orbit[x][46] = (
                f[46]
                - 3 * orbit[x][70]
                - 2 * orbit[x][68]
                - 2 * orbit[x][65]
                - 1 * orbit[x][63]
                - 1 * orbit[x][59]
            )
            orbit[x][45] = (
                f[45] - 2 * orbit[x][65] - 2 * orbit[x][62] - 3 * orbit[x][56]
            )
            orbit[x][44] = (f[44] - 1 * orbit[x][67] - 2 * orbit[x][61]) / 4
            orbit[x][43] = (
                f[43] - 2 * orbit[x][66] - 1 * orbit[x][60] - 1 * orbit[x][59]
            ) / 2
            orbit[x][42] = (
                f[42]
                - 2 * orbit[x][71]
                - 4 * orbit[x][69]
                - 2 * orbit[x][67]
                - 2 * orbit[x][61]
                - 3 * orbit[x][55]
            )
            orbit[x][41] = (
                f[41]
                - 2 * orbit[x][71]
                - 1 * orbit[x][68]
                - 2 * orbit[x][67]
                - 1 * orbit[x][60]
                - 3 * orbit[x][55]
            )
            orbit[x][40] = (
                f[40]
                - 6 * orbit[x][70]
                - 2 * orbit[x][68]
                - 2 * orbit[x][66]
                - 4 * orbit[x][65]
                - 1 * orbit[x][60]
                - 1 * orbit[x][59]
                - 4 * orbit[x][54]
            )
            orbit[x][39] = (
                f[39] - 4 * orbit[x][65] - 1 * orbit[x][59] - 6 * orbit[x][56]
            ) / 2
            orbit[x][38] = (
                f[38]
                - 1 * orbit[x][68]
                - 1 * orbit[x][64]
                - 2 * orbit[x][63]
                - 1 * orbit[x][53]
                - 3 * orbit[x][50]
            )
            orbit[x][37] = (
                f[37]
                - 2 * orbit[x][68]
                - 2 * orbit[x][64]
                - 2 * orbit[x][63]
                - 4 * orbit[x][62]
                - 1 * orbit[x][53]
                - 1 * orbit[x][51]
                - 4 * orbit[x][49]
            )
            orbit[x][36] = (
                f[36]
                - 1 * orbit[x][68]
                - 2 * orbit[x][63]
                - 2 * orbit[x][62]
                - 1 * orbit[x][51]
                - 3 * orbit[x][50]
            )
            orbit[x][35] = (
                f[35] - 1 * orbit[x][59] - 2 * orbit[x][52] - 2 * orbit[x][45]
            ) / 2
            orbit[x][34] = (
                f[34] - 1 * orbit[x][59] - 2 * orbit[x][52] - 1 * orbit[x][51]
            ) / 2
            orbit[x][33] = (
                f[33]
                - 1 * orbit[x][67]
                - 2 * orbit[x][61]
                - 3 * orbit[x][58]
                - 4 * orbit[x][44]
                - 2 * orbit[x][42]
            ) / 2
            orbit[x][32] = (
                f[32]
                - 2 * orbit[x][66]
                - 1 * orbit[x][60]
                - 1 * orbit[x][59]
                - 2 * orbit[x][57]
                - 2 * orbit[x][43]
                - 2 * orbit[x][41]
                - 1 * orbit[x][40]
            ) / 2
            orbit[x][31] = (
                f[31]
                - 2 * orbit[x][65]
                - 1 * orbit[x][59]
                - 3 * orbit[x][56]
                - 1 * orbit[x][43]
                - 2 * orbit[x][39]
            )
            orbit[x][30] = (
                f[30]
                - 1 * orbit[x][67]
                - 1 * orbit[x][63]
                - 2 * orbit[x][61]
                - 1 * orbit[x][53]
                - 4 * orbit[x][44]
            )
            orbit[x][29] = (
                f[29]
                - 2 * orbit[x][66]
                - 2 * orbit[x][64]
                - 1 * orbit[x][60]
                - 1 * orbit[x][59]
                - 1 * orbit[x][53]
                - 2 * orbit[x][52]
                - 2 * orbit[x][43]
            )
            orbit[x][28] = (
                f[28]
                - 2 * orbit[x][65]
                - 2 * orbit[x][62]
                - 1 * orbit[x][59]
                - 1 * orbit[x][51]
                - 1 * orbit[x][43]
            )
            orbit[x][27] = (
                f[27] - 1 * orbit[x][59] - 1 * orbit[x][51] - 2 * orbit[x][45]
            ) / 2
            orbit[x][26] = (
                f[26]
                - 2 * orbit[x][67]
                - 2 * orbit[x][63]
                - 2 * orbit[x][61]
                - 6 * orbit[x][58]
                - 1 * orbit[x][53]
                - 2 * orbit[x][47]
                - 2 * orbit[x][42]
            )
            orbit[x][25] = (
                f[25]
                - 2 * orbit[x][66]
                - 2 * orbit[x][64]
                - 1 * orbit[x][59]
                - 2 * orbit[x][57]
                - 2 * orbit[x][52]
                - 1 * orbit[x][48]
                - 1 * orbit[x][40]
            ) / 2
            orbit[x][24] = (
                f[24]
                - 4 * orbit[x][65]
                - 4 * orbit[x][62]
                - 1 * orbit[x][59]
                - 6 * orbit[x][56]
                - 1 * orbit[x][51]
                - 2 * orbit[x][45]
                - 2 * orbit[x][39]
            )
            orbit[x][23] = (
                f[23] - 1 * orbit[x][55] - 1 * orbit[x][42] - 2 * orbit[x][33]
            ) / 4
            orbit[x][22] = (
                f[22]
                - 2 * orbit[x][54]
                - 1 * orbit[x][40]
                - 1 * orbit[x][39]
                - 1 * orbit[x][32]
                - 2 * orbit[x][31]
            ) / 3
            orbit[x][21] = (
                f[21]
                - 3 * orbit[x][55]
                - 3 * orbit[x][50]
                - 2 * orbit[x][42]
                - 2 * orbit[x][38]
                - 2 * orbit[x][33]
            )
            orbit[x][20] = (
                f[20]
                - 2 * orbit[x][54]
                - 2 * orbit[x][49]
                - 1 * orbit[x][40]
                - 1 * orbit[x][37]
                - 1 * orbit[x][32]
            )
            orbit[x][19] = (
                f[19]
                - 4 * orbit[x][54]
                - 4 * orbit[x][49]
                - 1 * orbit[x][40]
                - 2 * orbit[x][39]
                - 1 * orbit[x][37]
                - 2 * orbit[x][35]
                - 2 * orbit[x][31]
            )
            orbit[x][18] = (
                f[18]
                - 1 * orbit[x][59]
                - 1 * orbit[x][51]
                - 2 * orbit[x][46]
                - 2 * orbit[x][45]
                - 2 * orbit[x][36]
                - 2 * orbit[x][27]
                - 1 * orbit[x][24]
            ) / 2
            orbit[x][17] = (
                f[17]
                - 1 * orbit[x][60]
                - 1 * orbit[x][53]
                - 1 * orbit[x][51]
                - 1 * orbit[x][48]
                - 1 * orbit[x][37]
                - 2 * orbit[x][34]
                - 2 * orbit[x][30]
            ) / 2
            orbit[x][16] = (
                f[16]
                - 1 * orbit[x][59]
                - 2 * orbit[x][52]
                - 1 * orbit[x][51]
                - 2 * orbit[x][46]
                - 2 * orbit[x][36]
                - 2 * orbit[x][34]
                - 1 * orbit[x][29]
            )
            orbit[x][15] = (
                f[15]
                - 1 * orbit[x][59]
                - 2 * orbit[x][52]
                - 1 * orbit[x][51]
                - 2 * orbit[x][45]
                - 2 * orbit[x][35]
                - 2 * orbit[x][34]
                - 2 * orbit[x][27]
            )

            # reset neigh_x
            neigh_x[self.degree_sorted_adj[x]] = False

        return orbit
