import numpy as np
from . import BoxCut
from . import BoxList
from collections import namedtuple, deque

Branch = namedtuple('branch', ('begin', 'end', 'depth'))


class Nested(BoxCut):

    def bic(self, unique_points, likelihood, dof):
        # k = n_parameters
        n_parameters = (2 * dof + 1)
        a = unique_points * np.log(likelihood / unique_points)
        b = n_parameters * np.log(unique_points)
        return a + b

    def extend_todo(self, boxes, depth=0, begin=0):
        for start, end in boxes.items():
            branch = Branch(start+begin, end+begin, depth)
            # print(branch)
            if end - start > 3:
                self.todo_que.append(branch)
            else:
                self.done_que.append(branch)

    def extend_done(self, boxes, depth=0, begin=0):
        for start, end in boxes.items():
            branch = Branch(start+begin, end+begin, depth)
            self.done_que.append(branch)

    def bic_of_boxes(self, boxes):
        n = boxes[-1]
        unique_points = (n**2 - n) / 2.
        # N = number of samples
        box_bic = self.bic(unique_points=unique_points,
                           likelihood=boxes.fitness,
                           dof=len(boxes))
        return box_bic

    def split_is_happening(self, unity, partition):
        union_fitness = unity.fitness
        split_fitness = partition.fitness
        # bic_union = union_fitness
        # bic_partitioned = split_fitness
        # TODO: adding this in messes with accuracy!
        bic_union = self.bic_of_boxes(unity)
        bic_partitioned  = self.bic_of_boxes(partition)
        # print('union', bic_union)
        # print('partitioned', bic_partitioned)
        # print(bic_partitioned < bic_union)
        # print()
        return bic_partitioned < bic_union

    def run(self):
        """ recursive BoxClustering Algorithm """
        matrix = self.orig
        N = len(matrix)
        self.N = N
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
                self.extend_todo(boxes=partition, depth=depth, begin=begin)
            else:
                choice = unity
            self.extend_done(boxes=choice, depth=depth, begin=begin)
            choice = BoxList([i + begin for i in choice])
        depths = {}
        for b in self.done_que:
            if b.depth not in depths:
                depths[b.depth] = []
            depths[b.depth].append(b.end)
        hboxes = []
        previous = set([])
        for depth, boxes in depths.items():
            current = set(boxes)
            current = current.union(previous)
            hbox = BoxList(sorted(current))
            # print(hbox)
            # hboxes[depth] = hbox
            hboxes.append(hbox)
            previous = current
        self.hboxes = hboxes
        return hboxes[-1]

    #     df = pd.DataFrame(self.done_que)
    #     self.df = df
    #     df2 = self.make_cluster_df(df)
    #     self.df2 = df2
    #     self.hboxes = self.make_hboxes(df2)

    # def make_cluster_df(self, df):
    #     begin = 0
    #     end = df['end'].max()
    #     max_depth = df['depth'].max()
    #     print(begin, end, max_depth)
    #     df2 = pd.DataFrame(index=range(begin, end), columns=(0, max_depth+1))
    #     for i, row in df.iterrows():
    #         df2.loc[range(row.begin, row.end), row.depth] = int(i)
    #     df2 = df2[[int(i) for i in range(0, max_depth+1)]]
    #     df3 = df2.T.fillna(method='ffill')
    #     df2 = df3.T
    #     return df2

    # def make_hboxes(self, df2):
    #     b = {}
    #     cols = sorted(set(df2.columns))
    #     print(cols)
    #     for col in cols:
    #         boxes_list = []
    #         print(col)
    #         for i in set(df2[col]):
    #             d = df2.loc[df2[col] == i, col].index
    #             if len(d):
    #                 boxes_list.append(d[-1] + 1)
    #         if self.N not in boxes_list:
    #             boxes_list.append(self.N)
    #         boxes_list = sorted(boxes_list)
    #         hboxes = BoxList(boxes_list)
    #         b[col] = hboxes
    #     return b
