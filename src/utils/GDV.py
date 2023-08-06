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
        if b in self.adj[a]:
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
        if n is None:
            n = max(edges.flatten()).item() + 1
        # edges = GDV.make_directed(edges)
        m = edges.shape[1]

        self.adj = []
        deg = []
        for i in range(n):
            neighbors = find_neighbors_(i, edges)
            self.adj.append(neighbors)
            deg.append(len(neighbors))

        # deg, adj, _, _ =  Graph.add_structural_features_(edges)
        inc = []
        for i in range(n):
            inc.append([])

        for i in range(m):
            a, b = edges[:, i].tolist()
            inc[a].append([b, i])
            # inc[b].append([a, i])

        common2 = dict()
        common3 = dict()
        orbit = np.zeros((n, 73), dtype=int)
        for x in range(n):
            for n1 in range(deg[x]):
                a = self.adj[x][n1]
                for n2 in range(n1 + 1, deg[x]):
                    b = self.adj[x][n2]
                    if not self.adjacent(a, b):
                        continue
                    ab = Pair(a, b, n)
                    if ab.key not in common2.keys():
                        common2[ab.key] = 1
                    else:
                        common2[ab.key] += 1

                    for n3 in range(n2 + 1, deg[x]):
                        c = self.adj[x][n3]
                        # st = (
                        #     # self.adjacent(a, b)
                        #     + self.adjacent(a, c)
                        #     + self.adjacent(b, c)
                        # )
                        # if st < 2:
                        #     continue
                        if not self.adjacent(a, c):
                            continue
                        if not self.adjacent(b, c):
                            continue

                        abc = Triple(a, b, c, n)
                        if abc.key not in common3.keys():
                            common3[abc.key] = 1
                        else:
                            common3[abc.key] += 1

        tri = m * [0]
        for i in range(m):
            x, y = edges[:, i].tolist()

            common_neighbors = np.intersect1d(self.adj[x], self.adj[y])
            tri[i] = len(common_neighbors)

        tri = [
            val / 2 for val in tri
        ]  # every triangle counts twice becase of (x,y) (y,x)

        C5 = n * [0]
        neigh = n * [0]
        neigh2 = n * [0]

        for x in range(n):
            for nx in range(deg[x]):
                y = self.adj[x][nx]
                if y >= x:
                    break

                nn = 0
                for ny in range(deg[y]):
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
                            neigh[nn2] = zz
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
        for x in range(n):
            for i in range(ncx):
                common_x[common_x_list[i]] = 0

            ncx = 0

            # smaller graphlets
            orbit[x][0] = deg[x]
            for nx1 in range(deg[x]):
                a = self.adj[x][nx1]
                for nx2 in range(nx1 + 1, deg[x]):
                    b = self.adj[x][nx2]
                    if self.adjacent(a, b):
                        orbit[x][3] += 1
                    else:
                        orbit[x][2] += 1

                for na in range(deg[a]):
                    b = self.adj[a][na]
                    if b != x and not self.adjacent(x, b):
                        orbit[x][1] += 1
                        if common_x[b] == 0:
                            common_x_list[ncx] = b
                            ncx += 1

                        common_x[b] += 1

            f_71 = 0
            f_70 = 0
            f_67 = 0
            f_66 = 0
            f_58 = 0
            f_57 = 0
            f_69 = 0
            f_68 = 0
            f_64 = 0
            f_61 = 0
            f_60 = 0
            f_55 = 0
            f_48 = 0
            f_42 = 0
            f_41 = 0
            f_65 = 0
            f_63 = 0
            f_59 = 0
            f_54 = 0
            f_47 = 0
            f_46 = 0
            f_40 = 0
            f_62 = 0
            f_53 = 0
            f_51 = 0
            f_50 = 0
            f_49 = 0
            f_38 = 0
            f_37 = 0
            f_36 = 0
            f_44 = 0
            f_33 = 0
            f_30 = 0
            f_26 = 0
            f_52 = 0
            f_43 = 0
            f_32 = 0
            f_29 = 0
            f_25 = 0
            f_56 = 0
            f_45 = 0
            f_39 = 0
            f_31 = 0
            f_28 = 0
            f_24 = 0
            f_35 = 0
            f_34 = 0
            f_27 = 0
            f_18 = 0
            f_16 = 0
            f_15 = 0
            f_17 = 0
            f_22 = 0
            f_20 = 0
            f_19 = 0
            f_23 = 0
            f_21 = 0

            for nx1 in range(deg[x]):
                a, xa = inc[x][nx1]

                for i in range(nca):
                    common_a[common_a_list[i]] = 0
                nca = 0

                for na in range(deg[a]):
                    b = self.adj[a][na]
                    for nb in range(deg[b]):
                        c = self.adj[b][nb]
                        if c == a or self.adjacent(a, c):
                            continue
                        if common_a[c] == 0:
                            common_a_list[nca] = c
                            nca += 1

                        common_a[c] += 1

                # x = orbit-14 (tetrahedron)
                for nx2 in range(nx1 + 1, deg[x]):
                    b, xb = inc[x][nx2]

                    if not self.adjacent(a, b):
                        continue
                    for nx3 in range(nx2 + 1, deg[x]):
                        c, xc = inc[x][nx3]
                        if not self.adjacent(a, c) or not self.adjacent(b, c):
                            continue
                        orbit[x][14] += 1
                        f_70 += common3.get(Triple(a, b, c, n), 0) - 1
                        if tri[xa] > 2 and tri[xb] > 2:
                            f_71 += common3.get(Triple(x, a, b, n).key, 0) - 1
                        if tri[xa] > 2 and tri[xc] > 2:
                            f_71 += common3.get(Triple(x, a, c, n).key, 0) - 1
                        if tri[xb] > 2 and tri[xc] > 2:
                            f_71 += common3.get(Triple(x, b, c, n).key, 0) - 1
                        f_67 += tri[xa] - 2 + tri[xb] - 2 + tri[xc] - 2
                        f_66 += common2.get(Pair(a, b, n).key, 0) - 2
                        f_66 += common2.get(Pair(a, c, n).key, 0) - 2
                        f_66 += common2.get(Pair(b, c, n).key, 0) - 2
                        f_58 += deg[x] - 3
                        f_57 += deg[a] - 3 + deg[b] - 3 + deg[c] - 3

                # x = orbit-13 (diamond)
                for nx2 in range(0, deg[x]):
                    b, xb = inc[x][nx2]

                    if not self.adjacent(a, b):
                        continue
                    for nx3 in range(nx2 + 1, deg[x]):
                        c, xc = inc[x][nx3]
                        if not self.adjacent(a, c) or self.adjacent(b, c):
                            continue
                        orbit[x][13] += 1
                        if tri[xb] > 1 and tri[xc] > 1:
                            f_69 += common3.get(Triple(x, b, c, n).key, 0) - 1
                        # f_68 += (
                        #     common3.get(Triple(a, b, c, n).key, 0) - 1
                        # )  # exception
                        f_68 += (
                            len(
                                np.intersect1d(
                                    np.intersect1d(
                                        self.adj[a], self.adj[b], assume_unique=True
                                    ),
                                    self.adj[c],
                                    assume_unique=True,
                                )
                            )
                            - 1
                        )
                        # f_64 += common2.get(Pair(b, c, n).key, 0) - 2  # exception
                        f_64 += (
                            len(
                                np.intersect1d(
                                    self.adj[b], self.adj[c], assume_unique=True
                                )
                            )
                            - 2
                        )
                        f_61 += tri[xb] - 1 + tri[xc] - 1
                        f_60 += common2.get(Pair(a, b, n).key, 0) - 1
                        f_60 += common2.get(Pair(a, c, n).key, 0) - 1
                        f_55 += tri[xa] - 2
                        f_48 += deg[b] - 2 + deg[c] - 2
                        f_42 += deg[x] - 3
                        f_41 += deg[a] - 3

                # x = orbit-13 (diamond)
                # for nx2 in range(nx1 + 1, deg[x]):
                #     b, xb = inc[x][nx2]

                #     if self.adjacent(a, b):
                #         continue
                #     for nx3 in range(0, deg[x]):
                #         c, xc = inc[x][nx3]
                #         if not self.adjacent(a, c) or not self.adjacent(b, c):
                #             continue
                #         orbit[x][13] += 1
                #         if tri[xb] > 1 and tri[xa] > 1:
                #             f_69 += common3.get(Triple(x, a, b, n).key, 0) - 1
                #         f_68 += common3.get(Triple(a, b, c, n).key, 0) - 1  # exception
                #         f_64 += common2.get(Pair(a, b, n).key, 0) - 2  # exception
                #         f_61 += tri[xb] - 1 + tri[xa] - 1
                #         f_60 += common2.get(Pair(a, c, n).key, 0) - 1
                #         f_60 += common2.get(Pair(b, c, n).key, 0) - 1
                #         f_55 += tri[xc] - 2
                #         f_48 += deg[a] - 2 + deg[b] - 2
                #         f_42 += deg[x] - 3
                #         f_41 += deg[c] - 3

                # x = orbit-12 (diamond)
                for nx2 in range(nx1 + 1, deg[x]):
                    b, xb = inc[x][nx2]

                    if not self.adjacent(a, b):
                        continue
                    for na in range(deg[a]):
                        c, ac = inc[a][na]
                        if c == x or self.adjacent(x, c) or not self.adjacent(b, c):
                            continue
                        orbit[x][12] += 1
                        if tri[ac] > 1:
                            f_65 += common3.get(Triple(a, b, c, n).key, 0)
                        f_63 += common_x[c] - 2
                        f_59 += tri[ac] - 1 + common2.get(Pair(b, c, n).key, 0) - 1
                        f_54 += common2.get(Pair(a, b, n).key, 0) - 2
                        f_47 += deg[x] - 2
                        f_46 += deg[c] - 2
                        f_40 += deg[a] - 3 + deg[b] - 3

                # x = orbit-8 (cycle)
                for nx2 in range(nx1 + 1, deg[x]):
                    b, xb = inc[x][nx2]

                    if self.adjacent(a, b):
                        continue
                    for na in range(deg[a]):
                        c, ac = inc[a][na]
                        if c == x or self.adjacent(x, c) or not self.adjacent(b, c):
                            continue
                        orbit[x][8] += 1
                        if tri[ac] > 0:
                            # f_62 += common3.get(Triple(a, b, c, n).key, 0)  # exception
                            f_62 += len(
                                np.intersect1d(
                                    np.intersect1d(
                                        self.adj[a], self.adj[b], assume_unique=True
                                    ),
                                    self.adj[c],
                                    assume_unique=True,
                                )
                            )
                        f_53 += tri[xa] + tri[xb]
                        f_51 += tri[ac] + common2.get(Pair(b, c, n).key, 0)
                        f_50 += common_x[c] - 2
                        f_49 += common_a[b] - 2
                        f_38 += deg[x] - 2
                        f_37 += deg[a] - 2 + deg[b] - 2
                        f_36 += deg[c] - 2

                # x = orbit-11 (paw)
                for nx2 in range(nx1 + 1, deg[x]):
                    b, xb = inc[x][nx2]

                    if not self.adjacent(a, b):
                        continue
                    for nx3 in range(0, deg[x]):
                        c, xc = inc[x][nx3]
                        if (
                            c == a
                            or c == b
                            or self.adjacent(a, c)
                            or self.adjacent(b, c)
                        ):
                            continue
                        orbit[x][11] += 1
                        f_44 += tri[xc]
                        f_33 += deg[x] - 3
                        f_30 += deg[c] - 1
                        f_26 += deg[a] - 2 + deg[b] - 2

                # x = orbit-10 (paw)
                for nx2 in range(0, deg[x]):
                    b, xb = inc[x][nx2]

                    if not self.adjacent(a, b):
                        continue
                    for nb in range(0, deg[b]):
                        c, bc = inc[b][nb]
                        if (
                            c == x
                            or c == a
                            or self.adjacent(a, c)
                            or self.adjacent(x, c)
                        ):
                            continue
                        orbit[x][10] += 1
                        f_52 += common_a[c] - 1
                        f_43 += tri[bc]
                        f_32 += deg[b] - 3
                        f_29 += deg[c] - 1
                        f_25 += deg[a] - 2

                # x = orbit-9 (paw)
                for na1 in range(0, deg[a]):
                    b, ab = inc[a][na1]

                    if b == x or self.adjacent(x, b):
                        continue
                    for na2 in range(na1 + 1, deg[a]):
                        c, ac = inc[a][na2]
                        if c == x or not self.adjacent(b, c) or self.adjacent(x, c):
                            continue
                        orbit[x][9] += 1
                        if tri[ab] > 1 and tri[ac] > 1:
                            f_56 += common3.get(Triple(a, b, c, n).key, 0)
                        f_45 += common2.get(Pair(b, c, n).key, 0) - 1
                        f_39 += tri[ab] - 1 + tri[ac] - 1
                        f_31 += deg[a] - 3
                        f_28 += deg[x] - 1
                        f_24 += deg[b] - 2 + deg[c] - 2

                # x = orbit-4 (path)
                for na in range(0, deg[a]):
                    b, ab = inc[a][na]

                    if b == x or self.adjacent(x, b):
                        continue
                    for nb in range(0, deg[b]):
                        c, bc = inc[b][nb]
                        if c == a or self.adjacent(a, c) or self.adjacent(x, c):
                            continue
                        orbit[x][4] += 1
                        f_35 += common_a[c] - 1
                        f_34 += common_x[c]
                        f_27 += tri[bc]
                        f_18 += deg[b] - 2
                        f_16 += deg[x] - 1
                        f_15 += deg[c] - 1

                # x = orbit-5 (path)
                for nx2 in range(0, deg[x]):
                    b, xb = inc[x][nx2]

                    if b == a or self.adjacent(a, b):
                        continue
                    for nb in range(0, deg[b]):
                        c, bc = inc[b][nb]
                        if c == x or self.adjacent(a, c) or self.adjacent(x, c):
                            continue
                        orbit[x][5] += 1
                        f_17 += deg[a] - 1

                # x = orbit-6 (claw)
                for na1 in range(0, deg[a]):
                    b, ab = inc[a][na1]

                    if b == x or self.adjacent(x, b):
                        continue
                    for na2 in range(na1 + 1, deg[a]):
                        c, ac = inc[a][na2]
                        if c == x or self.adjacent(x, c) or self.adjacent(b, c):
                            continue
                        orbit[x][6] += 1
                        f_22 += deg[a] - 3
                        f_20 += deg[x] - 1
                        f_19 += deg[b] - 1 + deg[c] - 1

                # x = orbit-7 (claw)
                for nx2 in range(nx1 + 1, deg[x]):
                    b, xb = inc[x][nx2]

                    if self.adjacent(a, b):
                        continue
                    for nx3 in range(nx2 + 1, deg[x]):
                        c, xc = inc[x][nx3]
                        if self.adjacent(a, c) or self.adjacent(b, c):
                            continue
                        orbit[x][7] += 1
                        f_23 += deg[x] - 3
                        f_21 += deg[a] - 1 + deg[b] - 1 + deg[c] - 1

            # solve equations
            orbit[x][72] = C5[x]
            orbit[x][71] = (f_71 - 12 * orbit[x][72]) / 2
            orbit[x][70] = f_70 - 4 * orbit[x][72]
            orbit[x][69] = (f_69 - 2 * orbit[x][71]) / 4
            orbit[x][68] = f_68 - 2 * orbit[x][71]
            orbit[x][67] = f_67 - 12 * orbit[x][72] - 4 * orbit[x][71]
            orbit[x][66] = (
                f_66 - 12 * orbit[x][72] - 2 * orbit[x][71] - 3 * orbit[x][70]
            )
            orbit[x][65] = (f_65 - 3 * orbit[x][70]) / 2
            orbit[x][64] = f_64 - 2 * orbit[x][71] - 4 * orbit[x][69] - 1 * orbit[x][68]
            orbit[x][63] = f_63 - 3 * orbit[x][70] - 2 * orbit[x][68]
            orbit[x][62] = (f_62 - 1 * orbit[x][68]) / 2
            orbit[x][61] = (
                f_61 - 4 * orbit[x][71] - 8 * orbit[x][69] - 2 * orbit[x][67]
            ) / 2
            orbit[x][60] = f_60 - 4 * orbit[x][71] - 2 * orbit[x][68] - 2 * orbit[x][67]
            orbit[x][59] = f_59 - 6 * orbit[x][70] - 2 * orbit[x][68] - 4 * orbit[x][65]
            orbit[x][58] = f_58 - 4 * orbit[x][72] - 2 * orbit[x][71] - 1 * orbit[x][67]
            orbit[x][57] = (
                f_57
                - 12 * orbit[x][72]
                - 4 * orbit[x][71]
                - 3 * orbit[x][70]
                - 1 * orbit[x][67]
                - 2 * orbit[x][66]
            )
            orbit[x][56] = (f_56 - 2 * orbit[x][65]) / 3
            orbit[x][55] = (f_55 - 2 * orbit[x][71] - 2 * orbit[x][67]) / 3
            orbit[x][54] = (
                f_54 - 3 * orbit[x][70] - 1 * orbit[x][66] - 2 * orbit[x][65]
            ) / 2
            orbit[x][53] = f_53 - 2 * orbit[x][68] - 2 * orbit[x][64] - 2 * orbit[x][63]
            orbit[x][52] = (
                f_52 - 2 * orbit[x][66] - 2 * orbit[x][64] - 1 * orbit[x][59]
            ) / 2
            orbit[x][51] = f_51 - 2 * orbit[x][68] - 2 * orbit[x][63] - 4 * orbit[x][62]
            orbit[x][50] = (f_50 - 1 * orbit[x][68] - 2 * orbit[x][63]) / 3
            orbit[x][49] = (
                f_49 - 1 * orbit[x][68] - 1 * orbit[x][64] - 2 * orbit[x][62]
            ) / 2
            orbit[x][48] = (
                f_48
                - 4 * orbit[x][71]
                - 8 * orbit[x][69]
                - 2 * orbit[x][68]
                - 2 * orbit[x][67]
                - 2 * orbit[x][64]
                - 2 * orbit[x][61]
                - 1 * orbit[x][60]
            )
            orbit[x][47] = (
                f_47
                - 3 * orbit[x][70]
                - 2 * orbit[x][68]
                - 1 * orbit[x][66]
                - 1 * orbit[x][63]
                - 1 * orbit[x][60]
            )
            orbit[x][46] = (
                f_46
                - 3 * orbit[x][70]
                - 2 * orbit[x][68]
                - 2 * orbit[x][65]
                - 1 * orbit[x][63]
                - 1 * orbit[x][59]
            )
            orbit[x][45] = f_45 - 2 * orbit[x][65] - 2 * orbit[x][62] - 3 * orbit[x][56]
            orbit[x][44] = (f_44 - 1 * orbit[x][67] - 2 * orbit[x][61]) / 4
            orbit[x][43] = (
                f_43 - 2 * orbit[x][66] - 1 * orbit[x][60] - 1 * orbit[x][59]
            ) / 2
            orbit[x][42] = (
                f_42
                - 2 * orbit[x][71]
                - 4 * orbit[x][69]
                - 2 * orbit[x][67]
                - 2 * orbit[x][61]
                - 3 * orbit[x][55]
            )
            orbit[x][41] = (
                f_41
                - 2 * orbit[x][71]
                - 1 * orbit[x][68]
                - 2 * orbit[x][67]
                - 1 * orbit[x][60]
                - 3 * orbit[x][55]
            )
            orbit[x][40] = (
                f_40
                - 6 * orbit[x][70]
                - 2 * orbit[x][68]
                - 2 * orbit[x][66]
                - 4 * orbit[x][65]
                - 1 * orbit[x][60]
                - 1 * orbit[x][59]
                - 4 * orbit[x][54]
            )
            orbit[x][39] = (
                f_39 - 4 * orbit[x][65] - 1 * orbit[x][59] - 6 * orbit[x][56]
            ) / 2
            orbit[x][38] = (
                f_38
                - 1 * orbit[x][68]
                - 1 * orbit[x][64]
                - 2 * orbit[x][63]
                - 1 * orbit[x][53]
                - 3 * orbit[x][50]
            )
            orbit[x][37] = (
                f_37
                - 2 * orbit[x][68]
                - 2 * orbit[x][64]
                - 2 * orbit[x][63]
                - 4 * orbit[x][62]
                - 1 * orbit[x][53]
                - 1 * orbit[x][51]
                - 4 * orbit[x][49]
            )
            orbit[x][36] = (
                f_36
                - 1 * orbit[x][68]
                - 2 * orbit[x][63]
                - 2 * orbit[x][62]
                - 1 * orbit[x][51]
                - 3 * orbit[x][50]
            )
            orbit[x][35] = (
                f_35 - 1 * orbit[x][59] - 2 * orbit[x][52] - 2 * orbit[x][45]
            ) / 2
            orbit[x][34] = (
                f_34 - 1 * orbit[x][59] - 2 * orbit[x][52] - 1 * orbit[x][51]
            ) / 2
            orbit[x][33] = (
                f_33
                - 1 * orbit[x][67]
                - 2 * orbit[x][61]
                - 3 * orbit[x][58]
                - 4 * orbit[x][44]
                - 2 * orbit[x][42]
            ) / 2
            orbit[x][32] = (
                f_32
                - 2 * orbit[x][66]
                - 1 * orbit[x][60]
                - 1 * orbit[x][59]
                - 2 * orbit[x][57]
                - 2 * orbit[x][43]
                - 2 * orbit[x][41]
                - 1 * orbit[x][40]
            ) / 2
            orbit[x][31] = (
                f_31
                - 2 * orbit[x][65]
                - 1 * orbit[x][59]
                - 3 * orbit[x][56]
                - 1 * orbit[x][43]
                - 2 * orbit[x][39]
            )
            orbit[x][30] = (
                f_30
                - 1 * orbit[x][67]
                - 1 * orbit[x][63]
                - 2 * orbit[x][61]
                - 1 * orbit[x][53]
                - 4 * orbit[x][44]
            )
            orbit[x][29] = (
                f_29
                - 2 * orbit[x][66]
                - 2 * orbit[x][64]
                - 1 * orbit[x][60]
                - 1 * orbit[x][59]
                - 1 * orbit[x][53]
                - 2 * orbit[x][52]
                - 2 * orbit[x][43]
            )
            orbit[x][28] = (
                f_28
                - 2 * orbit[x][65]
                - 2 * orbit[x][62]
                - 1 * orbit[x][59]
                - 1 * orbit[x][51]
                - 1 * orbit[x][43]
            )
            orbit[x][27] = (
                f_27 - 1 * orbit[x][59] - 1 * orbit[x][51] - 2 * orbit[x][45]
            ) / 2
            orbit[x][26] = (
                f_26
                - 2 * orbit[x][67]
                - 2 * orbit[x][63]
                - 2 * orbit[x][61]
                - 6 * orbit[x][58]
                - 1 * orbit[x][53]
                - 2 * orbit[x][47]
                - 2 * orbit[x][42]
            )
            orbit[x][25] = (
                f_25
                - 2 * orbit[x][66]
                - 2 * orbit[x][64]
                - 1 * orbit[x][59]
                - 2 * orbit[x][57]
                - 2 * orbit[x][52]
                - 1 * orbit[x][48]
                - 1 * orbit[x][40]
            ) / 2
            orbit[x][24] = (
                f_24
                - 4 * orbit[x][65]
                - 4 * orbit[x][62]
                - 1 * orbit[x][59]
                - 6 * orbit[x][56]
                - 1 * orbit[x][51]
                - 2 * orbit[x][45]
                - 2 * orbit[x][39]
            )
            orbit[x][23] = (
                f_23 - 1 * orbit[x][55] - 1 * orbit[x][42] - 2 * orbit[x][33]
            ) / 4
            orbit[x][22] = (
                f_22
                - 2 * orbit[x][54]
                - 1 * orbit[x][40]
                - 1 * orbit[x][39]
                - 1 * orbit[x][32]
                - 2 * orbit[x][31]
            ) / 3
            orbit[x][21] = (
                f_21
                - 3 * orbit[x][55]
                - 3 * orbit[x][50]
                - 2 * orbit[x][42]
                - 2 * orbit[x][38]
                - 2 * orbit[x][33]
            )
            orbit[x][20] = (
                f_20
                - 2 * orbit[x][54]
                - 2 * orbit[x][49]
                - 1 * orbit[x][40]
                - 1 * orbit[x][37]
                - 1 * orbit[x][32]
            )
            orbit[x][19] = (
                f_19
                - 4 * orbit[x][54]
                - 4 * orbit[x][49]
                - 1 * orbit[x][40]
                - 2 * orbit[x][39]
                - 1 * orbit[x][37]
                - 2 * orbit[x][35]
                - 2 * orbit[x][31]
            )
            orbit[x][18] = (
                f_18
                - 1 * orbit[x][59]
                - 1 * orbit[x][51]
                - 2 * orbit[x][46]
                - 2 * orbit[x][45]
                - 2 * orbit[x][36]
                - 2 * orbit[x][27]
                - 1 * orbit[x][24]
            ) / 2
            orbit[x][17] = (
                f_17
                - 1 * orbit[x][60]
                - 1 * orbit[x][53]
                - 1 * orbit[x][51]
                - 1 * orbit[x][48]
                - 1 * orbit[x][37]
                - 2 * orbit[x][34]
                - 2 * orbit[x][30]
            ) / 2
            orbit[x][16] = (
                f_16
                - 1 * orbit[x][59]
                - 2 * orbit[x][52]
                - 1 * orbit[x][51]
                - 2 * orbit[x][46]
                - 2 * orbit[x][36]
                - 2 * orbit[x][34]
                - 1 * orbit[x][29]
            )
            orbit[x][15] = (
                f_15
                - 1 * orbit[x][59]
                - 2 * orbit[x][52]
                - 1 * orbit[x][51]
                - 2 * orbit[x][45]
                - 2 * orbit[x][35]
                - 2 * orbit[x][34]
                - 2 * orbit[x][27]
            )

        return orbit
