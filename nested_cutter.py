import numpy as np
from . import BoxCut
from . import BoxList
import pandas as pd
from collections import namedtuple, deque


# def bic(unique_points, likelihood, dof):
#     return unique_points * np.log(likelihood) + (2 * dof + 1
#                                                  ) * np.log(unique_points)
# # box is a list + fitness...
# class Box(list):
#     def __init__(self, *args, **kwds):
#         list.__init__(self, *args, **kwds)
#         self.fitness = None
# def cluster_series_to_clust_list(label_series):
#     """ turn a list of lists cluster format into a pd.Series

#     input:
#         - cluster_series: pd.Series with cluster ids as values and
#                         names as keys
#     output:
#         - clust_list: a list of lists containing element ids.
#                 each list represents one cluster
#     """
#     l = label_series
#     clust_list = []
#     for c in sorted(l.unique()):
#         clust = l[l == c]
#         clust = list(clust.index)
#         clust_list.append(clust)
#     return clust_list
# def clust_list_to_cluster_series(clust_list, order=None):
#     """ turn a list of lists cluster format into a pd.Series
#     input:
#         - clust_list: a list of lists containing element ids.
#                 each list represents one cluster
#         - order: optional order of labels in resulting pd.Series
#                  must contain every element inside clust_list
#     output:
#         - label_series: pd.Series with cluster ids as values and
#                         names as keys
#     """
#     d = {}
#     for clust_id, clust in enumerate(clust_list):
#         for item_name in clust:
#             d[item_name] = clust_id
#     label_series = pd.Series(d)
#     if order is not None:
#         assert len(order) == len(label_series)
#         label_series = label_series.loc[order]
#     return label_series
# # TODO: recursive BoxClustering Algorithm
# def hierarchical_box_clustering(self, branch=None):
#     # branch = current subset of boxes?
#     if branch is None:
#         self.hierarchy = Box()
#         for i in range(self.matrix_size):
#             self.hierarchy.append(i)
#         branch = self.hierarchy
#     # subset of full matrix
#     matrix = self.current_matrix[branch, :][:, branch]
#     # initial guess is all nodes same box
#     branch.fitness = self._evaluate_box_fitness(
#         boxes=[len(matrix)], matrix=matrix)
#     # solve.
#     boxes, fitness = self.fit_boxes(matrix=matrix, )
#     # adjust boxes to actual positions?
#     boxes = [i + branch[0] for i in boxes]
#     # for every box in boxes,
#     hierarchy = Box()
#     for i, n in enumerate(boxes):
#         if i == 0:
#             b = Box()
#             b.extend(range(branch[0], n))
#             hierarchy.append(b)
#         else:
#             b = Box()
#             b.extend(range(boxes[i - 1], n))
#             hierarchy.append(b)
#     hierarchy.fitness = fitness
#     # lists in lists
#     # [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
#     if len(hierarchy) == 1:
#         return hierarchy
#     for i, branch in enumerate(hierarchy):
#         if len(branch) == 1:
#             continue
#         proposed = self.hierarchical_box_clustering(branch=branch)
#         # TODO verify this mess...
#         N = (len(branch)**2 + len(branch)) / 2.
#         bic_partitioned = bic(N, proposed.fitness, len(proposed))
#         bic_union = bic(N, branch.fitness, 1)
#         print(branch)
#         print(proposed)
#         print(bic_partitioned, bic_union)
#         print()
#         if bic_partitioned < bic_union:
#             hierarchy[i] = proposed[:]
#     return hierarchy


Branch = namedtuple('branch', ('begin', 'end', 'depth'))


class Nested(BoxCut):

    def bic(self, unique_points, likelihood, dof):
        a = unique_points * np.log(likelihood)
        b = (2 * dof + 1) * np.log(unique_points)
        return a + b

    def extend_todo(self, boxes, depth=0, begin=0):
        for start, end in boxes.items():
            branch = Branch(start+begin, end+begin, depth)
            if end - start > 3:
                self.todo_que.append(branch)

    def extend_done(self, boxes, depth=0, begin=0):
        for start, end in boxes.items():
            branch = Branch(start+begin, end+begin, depth)
            self.done_que.append(branch)

    def split_is_happening(self, unity, partition):
        union_fitness = unity.fitness
        split_fitness = partition.fitness
        # print(union_fitness, split_fitness,  split_fitness < union_fitness)
        # N = (len(branch)**2 + len(branch)) / 2.
        # bic_partitioned = bic(N, split_fitness, len(split_boxes))
        # bic_union = bic(N, union_fitness, 1)
        # print(bic_union, bic_partitioned, bic_partitioned < bic_union)
        bic_partitioned = split_fitness
        bic_union = union_fitness
        return bic_partitioned < bic_union

    def run(self):
        """ recursive BoxClustering Algorithm """
        matrix = self.orig
        N = len(matrix)
        # populate que
        self.todo_que = deque([])
        self.done_que = []
        unity = BoxList([N])
        self.extend_todo(boxes=unity)
        self.extend_done(boxes=unity)

        while len(self.todo_que):
            branch = self.todo_que.popleft()
            begin, end = branch.begin, branch.end
            depth = branch.depth + 1

            submatrix = matrix[begin:end, :][:, begin:end]

            # no further splitting
            unity = BoxList([len(submatrix)])
            self._evaluate_box_fitness(unity, submatrix)

            # best further split
            partition, _ = self.fit_boxes(matrix=submatrix)

            # best wins.
            if self.split_is_happening(unity, partition):
                choice = partition
#                 print('split', branch)
#                 print('into')
                self.extend_todo(boxes=partition, depth=depth, begin=begin)
#             else:
#                 print('done', branch)
#                 print('into')


            self.extend_done(boxes=choice, depth=depth, begin=begin)
            choice = BoxList([i + begin for i in choice])
            # print(choice)
            # print()

        df = pd.DataFrame(self.done_que)
        self.df = df
        df2 = self.make_cluster_df(df)
        self.df2 = df2
        self.hboxes = self.make_hboxes(df2)

    def make_cluster_df(self, df):
        begin = 0
        end = df['end'].max()
        max_depth = df['depth'].max()
        print(begin, end, max_depth)
        df2 = pd.DataFrame(index=range(begin, end), columns=(0, max_depth+1))
        for i, row in df.iterrows():
            df2.loc[range(row.begin, row.end), row.depth] = int(i)
        df2 = df2[[int(i) for i in range(0, max_depth+1)]]
        df3 = df2.T.fillna(method='ffill')
        df2 = df3.T
        return df2

    def make_hboxes(self, df2):
        b = {}
        cols = sorted(set(df2.columns))
        for col in cols:
            boxes_list = []
            print(col)
            for i in set(df2[col]):
                # print(i)
                d = df2.loc[df2[col] == i, col].index
                boxes_list.append(d[-1] + 1)

            boxes_list = sorted(boxes_list)
            hboxes = BoxList(boxes_list)
            b[col] = hboxes
        return b
