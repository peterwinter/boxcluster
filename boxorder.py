from collections import namedtuple
import numpy as np
import random
import math


class BaseFitMatrix(object):
    """ basic class functionality """

    def __init__(self, matrix, fitness=np.inf, order=None, weights=None):
        self.matrix = matrix
        self.n = len(matrix)
        self.fitness = fitness
        if order is None:
            order = np.arange(len(matrix), dtype=int)
        self.order = order
        if weights is None:
            weights = self.make_weight_matrix()
        self.weight_matrix = weights
        self.calculate_fitness()

    def make_weight_matrix(self):
        weight_matrix = np.ones(shape=self.matrix.shape)
        n = self.n
        for i in range(n):
            for j in range(n):
                weight_matrix[i, j] -= abs(i - j) / n
        return weight_matrix

    def calculate_fitness(self):
        """returns the fitness of the matrix, which is the product of the
        current adjacency matrix and the weight matrix"""
        fitness = (self.matrix * self.weight_matrix).sum()
        self.fitness = fitness
        return fitness

    def copy(self):
        return self.__class__(matrix=self.matrix.copy(),
                              fitness=self.fitness,
                              order=self.order.copy(),
                              weights=self.weight_matrix)

    def __len__(self):
        return len(self.matrix)

    def __repr__(self):
        f = round(self.fitness, ndigits=2)
        return 'Fit:{f} matrix: {m}'.format(f=f, m=self.matrix)

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __le__(self, other):
        return self.fitness <= other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __ne__(self, other):
        return self.fitness != other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __ge__(self, other):
        return self.fitness >= other.fitness

    def inverse_order(self):
        return np.argsort(self.order)

    # def __getitem__(self, key):
    #     return self.boxes[key]

    # def __setitem__(self, key, value):
    #     self.boxes[key] = value

    # def __delitem__(self, key):
    #     del self.boxes[key]
    # def insert(self, key, value):
    #     self.boxes.insert(key, value)


class BaseSortMatrix(BaseFitMatrix):

    def origional_matrix(self):
        inverse = self.inverse_order()
        return self.matrix[inverse, :][:, inverse]

    def reorder(self, order):
        """put the matrix in a new order"""
        order = np.array(order)
        self.matrix = self.matrix[order, :]
        self.matrix = self.matrix[:, order]
        self.order = self.order[order]
        self.calculate_fitness()
        return self.matrix

    def smart_reorder(self, order, matrix_range):
        """reorder the matrix, but do it a faster way. most moves will only
        affect a small fraction of the matrix, so there's no reason to move
        most of the elements"""
        matrix = self.matrix
        # .copy()
        cautious = True
        if cautious:
            msg = 'order and matrix range variables must contain same contents'
            msg = '{msg}:{o}\n:{mr}'.format(msg=msg, o=order, mr=matrix_range)
            assert set(order) == set(matrix_range), msg
        start = matrix_range[0]
        stop = matrix_range[-1] + 1
        # sub_matrix = self.current_matrix[order, :]
        sub_matrix = matrix[order, :]
        matrix[start:stop, :] = sub_matrix
        sub_matrix = matrix[:, order]
        matrix[:, start:stop] = sub_matrix
        self.calculate_fitness()
        self.matrix = matrix
        self.order[start:stop] = order
        return matrix

    def shuffle(self):
        random_order = self.order
        np.random.shuffle(random_order)
        self.reorder(random_order)

class OrderedArray(BaseSortMatrix):

    def determine_size(self):
        """size is random integer from pareto distribution"""
        size = np.inf
        while size >= self.n:
            size = np.random.pareto(0.2)
            size = int(math.ceil(size))
        return size

    def determine_positions(self, size):
        # roll the dice for the location---the starting position of the slice
        position = random.randrange(0, self.n - size)
        # TODO: size should probably depend on the temp!
        while 1:
            new_pos = np.random.pareto(0.2)
            new_pos = int(math.ceil(new_pos))
            # random sign
            if np.random.random() < 0.5:
                new_pos = -new_pos
            if 0 <= (position + new_pos) <= (self.n - size):
                break
        return position, new_pos

    def propose_cuts(self, size, position, new_pos):
        Cuts = namedtuple('cuts', ("lower_limit", "lower_cut", "pivot",
                                   "upper_cut", "upper_limit"))
        # the lowest and highest positions
        lower_limit = 0
        upper_limit = self.n
        # the upper edge of the origional box is the center
        # of movement for both up and down shifts.
        pivot = position + size
        if new_pos > 0:
            lower_cut = position
            upper_cut = position + size + new_pos
        elif new_pos < 0:
            lower_cut = position + new_pos
            upper_cut = position + size
        return Cuts(lower_limit, lower_cut, pivot, upper_cut, upper_limit)

    def cuts_to_order(self, cuts):
        # lower cut to pivot --> move right.
        # move left <-- pivot to upper cut.
        order = list(range(self.n))
        a = order[cuts.lower_limit:cuts.lower_cut]
        b = order[cuts.pivot:cuts.upper_cut]
        c = order[cuts.lower_cut:cuts.pivot]
        d = order[cuts.upper_cut:cuts.upper_limit]
        new_order = a + b + c + d
        return new_order

    def cuts_to_sub_matrix_range(self, cuts):
        sub_matrix_range = list(range(cuts.lower_cut, cuts.upper_cut))
        return sub_matrix_range

    def cuts_to_sub_order(self, cuts):
        move_right = list(range(cuts.lower_cut, cuts.pivot))
        move_left = list(range(cuts.pivot, cuts.upper_cut))
        sub_order = move_left + move_right
        return sub_order

    def propose_move(self):
        size = self.determine_size()
        position, new_pos = self.determine_positions(size)
        cuts = self.propose_cuts(size, position, new_pos)
        new_order = self.cuts_to_order(cuts)
        sub_matrix_range = self.cuts_to_sub_matrix_range(cuts)
        sub_order = self.cuts_to_sub_order(cuts)
        try:
            new_matrix = self.smart_reorder(sub_order, sub_matrix_range)
        except:
            print(position, new_pos, size, self.n)
            raise
        return new_matrix, new_order

# class BoxOrder(object):
#     def __init__(self, order):
#         "docstring"
#         self.order = order
#         self.Cuts = namedtuple('cuts', ("lower_limit", "lower_cut", "pivot",
#                                         "upper_cut", "upper_limit"))

#     def propose_cuts(self, size, position, new_pos):
#         # order = list(range(self.matrix_size))
#         order = np.arange(self.n)
#         # propose a new order

#         # the lowest and highest positions
#         lower_limit = 0
#         upper_limit = self.matrix_size

#         # the upper edge of the origional box is the center
#         # of movement for both up and down shifts.
#         pivot = position + size

#         if new_pos > 0:
#             lower_cut = position
#             upper_cut = position + size + new_pos

#         elif new_pos < 0:
#             lower_cut = postion + new_pos
#             upper_cut = position + size
#         return self.Cuts(lower_limit, lower_cut, pivot, upper_cut, upper_limit)

#     def cuts_to_order(self, cuts):
#         (lower_limit, lower_cut, pivot, upper_cut, upper_limit) = cuts
#         # lower cut to pivot --> move right.
#         # move left <-- pivot to upper cut.
#         order = np.arange(self.n)
#         a = order[lower_limit:lower_cut]
#         b = order[pivot:upper_cut]
#         c = order[lower_cut:pivot]
#         d = order[upper_cut:upper_limit]
#         new_order = a + b + c + d
#         return new_order

#     def cuts_to_sub_matrix_range(self, cuts):
#         (lower_limit, lower_cut, pivot, upper_cut, upper_limit) = cuts
#         sub_matrix_range = list(range(lower_cut, upper_cut))
#         return sub_matrix_range

#     def cuts_to_sub_matrix_range(self, cuts):
#         (lower_limit, lower_cut, pivot, upper_cut, upper_limit) = cuts
#         move_right = list(range(lower_cut, pivot))
#         move_left = list(range(pivot, upper_cut))
#         sub_order = move_left + move_right
#         return sub_order
