#!/usr/bin/env python
"""structures for performing a couple of different clustering algorithms. most
notably, performs matrix sorting as described in marta's pnas, using simulated
annealing"""

import numpy as np
import random
import math
# from scipy.stats import expon
from collections import namedtuple


class SortingAlgorithm(object):
    """base class for sorting algorithms. this establishes the
    weight matrix, and contains methods for moving things around
    in the matrix, but none of the meat-and-potatoes of the
    algorithm."""

    def __init__(self, original_matrix):
        self.matrix_size = len(original_matrix)
        self.orig = original_matrix
        self.current_matrix = self.orig
        # build a weight matrix that falls away from the diagonal this will be
        # useful later when
        self.weight_matrix = np.ones(self.orig.shape)
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                self.weight_matrix[i, j] -= (abs(i - j) /
                                             float(self.matrix_size))

    def reorder(self, order, matrix=None):
        """put the matrix in a new order"""
        if matrix is None:
            matrix = self.current_matrix
        return matrix[order, :][:, order]

    def smart_reorder(self, order, matrix_range, matrix=None):
        """reorder the matrix, but do it a faster way. most moves will only
        affect a small fraction of the matrix, so there's no reason to move
        most of the elements"""
        if matrix is None:
            matrix = self.current_matrix.copy()
        else:
            matrix = matrix.copy()
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
        return matrix

    def test_fitness(self, array=None):
        """returns the fitness of the matrix, which is the product of the
        current adjacency matrix and the weight matrix"""
        if array is None:
            array = self.current_matrix
        return (array * self.weight_matrix).sum()

# class HierarchicalClustering(SortingAlgorithm):
# from scipy.cluster import hierarchy
#     """use scipy.cluster.hierarchy to sort the matrix"""
#     def __call__(self):
#         self.order = hierarchy.leaves_list(hierarchy.linkage(self.orig))
#         self.result = self.reorder(self.order, self.orig)
#         return self.result, self.order


class BoxSort(SortingAlgorithm):
    def __init__(self, *args, **kwds):
        SortingAlgorithm.__init__(self, *args, **kwds)
        # set the counters and empty the files
        self._evals = 0
        self._last_move = 0
        self._last_best = 0
        self.current_matrix = self.orig
        self.current_order = np.arange(len(self.orig))
        self.current_fitness = self.test_fitness(self.current_matrix)

    def shuffle_order(self):
        order = self.current_order
        random.shuffle(order)
        self.current_order = np.array(order)
        self.current_matrix = self.reorder(self.current_order)

    def initialize(self):
        self.shuffle_order()
        # store best
        self.best_matrix = self.current_matrix
        self.best_order = self.current_order
        self.best_fitness = self.test_fitness(self.current_matrix)
        # initial state
        self.current_fitness = self.test_fitness(self.current_matrix)
        # constants
        # this heuristic seems weird:
        steps = int(self.matrix_size * (100 + self.matrix_size) / 100)
        self.steps = steps
        # keep track of the time since the last move
        self.last_move_time = 0
        self.finishing_threshold = self.finishing_criterion * steps
        self._delta = 0.  #how close we are to quitting

    def __call__(self,
                 cooling_factor=0.999,
                 temperature=0.001,
                 finishing_criterion=1,
                 verbosity=False):
        """this is where the sausage gets made.

Perform Simulated annealling to optimize the order of the matrix, which gets
saved to the file 'best_order.txt'. N move attempts are made at each
temperature, where N is the matrix order. This heuristic scales the algorithm
to the square of the number of nodes.

    parameters:
    cooling factor (default: 0.98)
        the principle determinant of the cooling schedule. if you set this too
        low, they system will freeze too fast.
    temperature (default: 0.01)
        the initial temperature of the system. its a good idea to leave this
        high---the cooling rate will increase if too many moves are being
        accepted.
    finishing_criterion (default: 1)
        the number of unproductive cooling steps to before we assume the system
        is frozen. practically, there is a finite probability that we're not in
        a global optimum. as the system cools, the probability of finding a
        better solution decreases. this parameter defines how long (how many
        cooling steps) we're willing to spin the wheels before accepting a
        solution.
    verbose (default: False)
        how much shit to print out
"""
        self.cooling_factor = cooling_factor
        self.temperature = temperature
        self.finishing_criterion = finishing_criterion
        self.verbosity = verbosity
        # always keep initialize after these assignments
        self.initialize()

        # each loop is a temperature
        while 1:
            # start a new temperature
            fraction_moved = self.temperature_movement_block()
            # if it has been a couple of cooling cycles since we last
            # improved, then return. this should be long enough to search
            # the local landscape and find the local maximum.
            if self.last_move_time > self.finishing_threshold:
                return self.best_matrix, self.best_order
            if fraction_moved > 0.1:
                self.temperature *= self.cooling_factor**100
            else:
                c_f = fraction_moved * 1000.
                self.temperature *= self.cooling_factor**int(np.ceil(c_f))

    def temperature_movement_block(self):
        self._moves_this_temp = 0
        # perform block of moves
        for n in range(self.steps):
            self.turn()
            self.last_move_time = self._evals - self._last_move
        # keep track of the number of productive moves at this
        # temperature so we can adaptively change the temp
        fraction_moved = float(self._moves_this_temp) / float(self.steps)
        return fraction_moved

    def determine_size(self):
        """size is random integer from pareto distribution"""
        size = np.inf
        while size >= self.matrix_size:
            size = np.random.pareto(0.2)
            size = int(math.ceil(size))
        return size

    def determine_positions(self, size):
        # roll the dice for the location---the starting position of the slice
        position = random.randrange(0, self.matrix_size - size)
        # TODO: size should probably depend on the temp!
        while 1:
            new_pos = np.random.pareto(0.2)
            new_pos = int(math.ceil(new_pos))
            # random sign
            if np.random.random() < 0.5:
                new_pos = -new_pos
            if 0 <= (position + new_pos) <= (self.matrix_size - size):
                break
        return position, new_pos

    def propose_cuts(self, size, position, new_pos):
        Cuts = namedtuple('cuts', ("lower_limit", "lower_cut", "pivot",
                                   "upper_cut", "upper_limit"))
        # the lowest and highest positions
        lower_limit = 0
        upper_limit = self.matrix_size
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
        order = list(range(self.matrix_size))
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

    def move_is_accepted(self, cur_fit, new_fit):
        # if roll some dice to find out if we accept the move
        if new_fit > cur_fit:
            return True
        else:
            improvement_ratio = (new_fit - cur_fit) / cur_fit
            p = np.exp(improvement_ratio / self.temperature)
        # if self.verbosity == 2:
        #     a = (self._evals, self.temperature, self.current_fitness, fitness,
        #          p, self._evals - self._last_move, size, position, new_pos)
        #     s = '%6d %.5e %.12e %.8e %.3e\t%5d %5d %5d %+5d' % a
        #     print(s, file=sys.stderr)
        return np.random.random() < p

    def propose_new_matrix(self):
        size = self.determine_size()
        position, new_pos = self.determine_positions(size)
        cuts = self.propose_cuts(size, position, new_pos)
        new_order = self.cuts_to_order(cuts)
        sub_matrix_range = self.cuts_to_sub_matrix_range(cuts)
        sub_order = self.cuts_to_sub_order(cuts)
        try:
            new_matrix = self.smart_reorder(sub_order, sub_matrix_range)
        except:
            print(position, new_pos, size, self.matrix_size)
            raise
        return new_matrix, new_order

    def turn(self):
        self._evals += 1
        new_matrix, new_order = self.propose_new_matrix()
        # calculate the probability of accepting the new order
        new_fit = self.test_fitness(new_matrix)
        cur_fit = self.current_fitness
        if self.move_is_accepted(cur_fit, new_fit):
            # protect against the kind of fruitless cycling that can occur if
            # two configurations have the same fitness
            if cur_fit != new_fit:
                self._last_move = self._evals
                self._moves_this_temp += 1
            # update matrix
            self.current_fitness = new_fit
            self.current_matrix = new_matrix
            self.current_order = self.current_order[new_order]
            self.update_status()
        # how close we are to quitting
        self._delta = self.last_move_time / self.finishing_threshold
        return new_fit

    def update_status(self):
        if self.current_fitness > self.best_fitness:
            # if self.verbosity == 1:
            #     a = (self._evals, self.temperature, \
            #             self.current_fitness, new_fit, p, \
            #             self._evals - self._last_move, size, position,\
            #             new_pos)
            #     print('%6d %.5e %.12e %.8e %.3e\t%5d %5d %5d %+5d' % a)
            self._last_best = self._evals
            self.best_fitness = self.current_fitness
            self.best_order = self.current_order
            self.best_matrix = self.current_matrix
