#!/usr/bin/env python
"""structures for performing a couple of different clustering algorithms. most
notably, performs matrix sorting as described in marta's pnas, using simulated
annealing"""

import sys
import numpy as np
import random
import math
from scipy.stats import expon
from scipy.cluster import hierarchy


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


class HierarchicalClustering(SortingAlgorithm):
    """use scipy.cluster.hierarchy to sort the matrix"""

    def __call__(self):
        self.order = hierarchy.leaves_list(hierarchy.linkage(self.orig))
        self.result = self.reorder(self.order, self.orig)
        return self.result, self.order


class BoxSort(SortingAlgorithm):

    def __init__(self, *args, **kwds):
        SortingAlgorithm.__init__(self, *args, **kwds)
        # set the counters and empty the files
        self._evals = 0
        self._last_move = 0
        self._last_best = 0
        # keep track of these in case something gets interrupted
        # open('sort_save.txt', 'w').close()
        # open('fit_save.txt', 'w').close()
        self.current_matrix = self.orig
        self.current_order = np.arange(len(self.orig))
        # log initial configuration
        self.current_fitness = self.test_fitness(self.current_matrix)
        # with open('initial_fitness.txt', 'w') as f:
        #     print(self.current_fitness, file=f)
        # shuffle the matrix and initialize tracker variables

    def initial_shuffle(self):
        order = self.current_order
        random.shuffle(order)
        self.current_order = np.array(order)
        self.current_matrix = self.reorder(self.current_order)

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
        self.initial_shuffle()
        # store best
        self.best_matrix = self.current_matrix
        self.best_order = self.current_order
        self.best_fitness = self.test_fitness(self.current_matrix)
        # initial state
        self.current_fitness = self.test_fitness(self.current_matrix)
        #
        self.temperature = temperature
        self.cooling_factor = cooling_factor
        self.finishing_criterion = finishing_criterion
        self._delta = 0.
        # constants
        self.verbosity = verbosity
        steps = int(self.matrix_size * (100 + self.matrix_size) / 100)
        self.steps = steps
        # keep track of the time since the last move
        last_move_time = 0
        # with open('best_order.txt', 'w') as h:
        # each loop is a temperature
        while 1:
            # start a new temperature
            self._moves_this_temp = 0
            for n in range(steps):

                # keep track of the number of productive moves at this
                # temperature so we can adaptively change the temp

                self.turn()
                last_move_time += self._evals - self._last_move

            # for i in self.best_order:
            #     print(i, file=h)

            fraction_moved = float(self._moves_this_temp) / float(steps)
            # since_last_best = self._evals - self._last_best

            # print(self.temperature,
            #       fraction_moved,
            #       self.best_fitness,
            #       np.ceil(1000. * fraction_moved),
            #       self._evals - self._last_move,
            #       steps * self.finishing_criterion)

            # self._save()
            # if it has been a couple of cooling cycles since we last
            # improved, then return. this should be long enough to search
            # the local  landscape and find the local maximum.
            if self._evals - self._last_move > steps * self.finishing_criterion:
                return self.best_matrix, self.best_order

            if fraction_moved > 0.1:
                self.temperature *= self.cooling_factor**100
            else:
                c_f = fraction_moved * 1000.
                self.temperature *= self.cooling_factor**int(np.ceil(c_f))

            # if since_last_best > steps * self.finishing_criterion:
            #    self.temperature *= self.cooling_factor ** 5

    # def _save(self):
    #     """cleanup that occurs at the end of a cooling step"""
    #     sys.stdout.flush()
    #     with open('fit_save.txt', 'a') as fit_handle:
    #         print((self.current_fitness, self.best_fitness), file=fit_handle)

    def turn(self):

        self._evals += 1

        # roll the dice for the size of the slice
        size = np.inf
        while size >= self.matrix_size:
            size = np.random.pareto(0.2)
            # size = expon.rvs(0., self.matrix_size/5.)*(1. - self._delta)
            if size < 1:
                size = 1  # for some reason, this was returning negative values
            size = int(math.ceil(size))

        # roll the dice for the location---the starting position of the slice
        position = random.randrange(0, self.matrix_size - size)

        # move it a distance
        # XXX this should probably depend on the temp!
        while 1:
            # new_pos = int(math.ceil(np.random.pareto(0.7)))
            new_pos = int(math.ceil(expon.rvs(0., self.matrix_size / 5.)))
            new_pos = np.random.pareto(0.2)
            new_pos = int(math.ceil(new_pos))

            # random sign
            if np.random.random() < 0.5:
                new_pos = -new_pos

            if 0 <= (position + new_pos) <= (self.matrix_size - size):
                break

        order = list(range(self.matrix_size))

        # propose a new order
        if new_pos > 0:
            new_order = order[0: position] + \
                        order[position+size: position+size+new_pos] + \
                        order[position: position+size] + \
                        order[position+size+new_pos: self.matrix_size]
            sub_matrix_range = list(range(position, position + size + new_pos))
            sub_order = list(range(position+size, position+size+new_pos)) + \
                        list(range(position, position+size))

        elif new_pos < 0:
            new_order = order[0: position+new_pos] + \
                        order[position: position+size] + \
                        order[position+new_pos: position] + \
                        order[position+size: self.matrix_size]

            sub_matrix_range = list(range(position + new_pos, position + size))
            sub_order = list(range(position, position+size)) + \
                        list(range(position+new_pos, position))

        #XXX
        # the proposed matrix and fitness that go with the new order
        try:
            new_matrix = self.smart_reorder(sub_order, sub_matrix_range)
        except:
            print(position, new_pos, size, self.matrix_size)
            raise

        fitness = self.test_fitness(new_matrix)

        # calculate the probability of accepting the new order
        cur_fit = self.current_fitness

        if fitness > cur_fit:
            p = 1
        else:
            p = np.exp((fitness - cur_fit) / cur_fit / self.temperature)

        if self.verbosity == 2:
            a = (self._evals, self.temperature, self.current_fitness, fitness,
                 p, self._evals - self._last_move, size, position, new_pos)
            s = '%6d %.5e %.12e %.8e %.3e\t%5d %5d %5d %+5d' % a
            print(s, file=sys.stderr)

        # if roll some dice to find out if we accept the move
        if np.random.random() < p:
            # protect against the kind of fruitless cycling that can occur if
            # two configurations have the same fitness
            if cur_fit != fitness:
                self._last_move = self._evals
                self._moves_this_temp += 1

            # update matrix
            self.current_fitness = fitness
            self.current_matrix = new_matrix
            self.current_order = self.current_order[new_order]

            if self.current_fitness > self.best_fitness:
                if self.verbosity == 1:
                    a = (self._evals, self.temperature, \
                         self.current_fitness, fitness, p, \
                         self._evals - self._last_move, size, position,\
                         new_pos)
                    print('%6d %.5e %.12e %.8e %.3e\t%5d %5d %5d %+5d' % a)

                self._last_best = self._evals
                self.best_fitness = self.current_fitness
                self.best_order = self.current_order
                self.best_matrix = self.current_matrix

        # how close we are to quitting
        self._delta = (self._evals - self._last_move
                       ) / self.steps / self.finishing_criterion
        return fitness
