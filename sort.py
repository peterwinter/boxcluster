#!/usr/bin/env python

import sys
import random
import math

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy


class ClusterMatrix(np.ndarray):
    def __new__(cls, data, *args, **kwargs):

        a = np.array(data).view(cls)

        a.hierarchical_clustering = HierarchicalClustering(a)
        a.deep_sort = DeepSort(a)

        return a


class SortingAlgorithm(object):
    def __init__(self, original_matrix):

        self.matrix_size = len(original_matrix)
        self.orig = original_matrix
        #self.order = range(len(original_matrix))
        #self.result = original_matrix

        # build a weight matrix that falls away from the diagonal
        self.weight_matrix = np.ones(self.orig.shape)
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                self.weight_matrix[i, j] -= (abs(i - j) /
                                             float(self.matrix_size))**0.5

    def reorder(self, order, matrix=None):
        if matrix == None:
            matrix = self.current_matrix
        #return np.transpose(matrix[order])[order]
        return matrix[order, :][:, order]

    def smart_reorder(self, order, matrix_range, matrix=None):
        if matrix == None:
            matrix = self.current_matrix.copy()
        else:
            matrix = matrix.copy()
        start = matrix_range[0]
        stop = matrix_range[-1] + 1
        sub_matrix = self.current_matrix[order, :]
        matrix[start:stop, :] = sub_matrix

        sub_matrix = matrix[:, order]
        matrix[:, start:stop] = sub_matrix

        return matrix

    #def test_fitness_crop(self, array=None):
    #    if array == None:
    #        array = self.current_matrix
    #    array_size = len(array)
    #    return (array * self.weight_matrix[:array_size,:array_size]).sum()

    def test_fitness(self, array=None):
        if array == None:
            array = self.current_matrix

        return (array * self.weight_matrix).sum()


class HierarchicalClustering(SortingAlgorithm):
    def __call__(self):

        self.order = hierarchy.leaves_list(hierarchy.linkage(self.orig))
        self.result = self.reorder(self.order, self.orig)
        return self.result, self.order


class DeepSort(SortingAlgorithm):
    def __init__(self, *args, **kwds):

        SortingAlgorithm.__init__(self, *args, **kwds)

        # set the counters and empty the files
        self._evals = 0
        self._last_move = 0
        self._last_best = 0

        open('sort_save.txt', 'w').close()
        open('fit_save.txt', 'w').close()

        self.current_matrix = self.orig
        self.current_order = np.arange(len(self.orig))

        # log initial configuration
        self.current_fitness = self.test_fitness(self.current_matrix)
        with open('initial_fitness.txt', 'w') as f:
            print(self.current_fitness, file=f)

        self.initial_shuffle()
        self.current_fitness = self.test_fitness(self.current_matrix)

        self.best_matrix = self.current_matrix
        self.best_order = self.current_order
        self.best_fitness = self.test_fitness(self.current_matrix)

    def initial_shuffle(self):
        order = self.current_order
        random.shuffle(order)
        self.current_order = np.array(order)

        self.current_matrix = self.reorder(self.current_order)

    def __call__(self,
                 cooling_factor=0.98,
                 temperature=0.0001,
                 finishing_criterion=5,
                 verbose=True):

        self.verbose = verbose
        self.temperature = temperature
        self.cooling_factor = cooling_factor
        self.finishing_criterion = finishing_criterion

        steps = int(self.matrix_size**2 / 100)

        # for small matrices
        if steps < 10:
            steps = 10

        # keep track of the time since the last move
        last_move_time = 0

        with open('best_order.txt', 'w') as h:
            while 1:
                for n in range(steps):
                    self.turn()
                    last_move_time += self._evals - self._last_move

                for i in self.best_order:
                    print(i, file=h)

                # the average time between accepted moves #XXX not what it says it
                # is
                mean_climb = last_move_time / float(steps) / 2
                since_last_best = self._evals - self._last_best

                self._save()
                # if it has been a couple of cooling cycles since we last improved,
                # then return. this should be long enough to search the local
                # landscape and find the local maximum.
                if self._evals - self._last_move > steps * self.finishing_criterion:
                    return self.best_matrix, self.best_order

                # adaptively anneal the system
                if mean_climb < 1:
                    self.temperature *= self.cooling_factor**20
                elif mean_climb < 5.:
                    self.temperature *= self.cooling_factor**10
                elif mean_climb < 10.:
                    self.temperature *= self.cooling_factor**5
                elif mean_climb < 20.:
                    self.temperature *= self.cooling_factor**5
                elif mean_climb < 50.:
                    self.temperature *= self.cooling_factor**2
                else:
                    self.temperature *= self.cooling_factor

                if since_last_best > steps * self.finishing_criterion:
                    self.temperature *= self.cooling_factor**5

    def _save(self):

        with open('fit_save.txt', 'a') as fit_handle:
            print((self.current_fitness, self.best_fitness), file=fit_handle)

        with open('sort_save.txt', 'a') as handle:
            for n, i in enumerate(self.best_order):
                if not n % 64:
                    print('\n=====', file=handle)
                    print('\n=====', file=handle)
                elif not n % 16:
                    print('\n---', file=handle)
                elif not n % 4:
                    print('\n-', file=handle)
                print('{i}\t'.format(i=i), file=handle)
            print('\n\n', file=handle)

        #matrix = Matrix(data=self.current_matrix,
        #                color_scheme='YlOrRd',
        #                domain=(.2, 1.),
        #                                )

        #matrix.write_file('current_matrix.agr')

    def turn(self):

        self._evals += 1

        # roll the dice for the size of the slice
        size = np.inf
        while size >= self.matrix_size:
            size = np.random.pareto(0.7)
            size = int(math.ceil(size))

        # roll the dice for the location---the starting position of the slice
        position = random.randrange(0, self.matrix_size - size)

        # move it a distance that depends on the temp
        while 1:
            new_pos = int(math.ceil(np.random.pareto(0.7)))

            # random sign
            if np.random.random() < 0.5:
                new_pos = -new_pos

            if 0 <= (position + new_pos) <= (self.matrix_size - size):
                break

        #print
        #print self.current_order
        #print self.current_matrix[0]
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

            #print new_order - np.arange(len(new_order))
            #print sub_matrix_range
            #print sub_order

        # the proposed matrix and fitness that go with the new order
        new_matrix = self.smart_reorder(sub_order, sub_matrix_range)
        fitness = self.test_fitness(new_matrix)

        # calculate the probability of accepting the new order
        cur_fit = self.current_fitness
        p = np.exp((fitness - cur_fit) / cur_fit / self.temperature)
        if self.verbose:
            s ='%6d %.5e %.12e %.8e %.3e\t%5d %5d %5d %+5d'%(self._evals, self.temperature, \
                                     self.current_fitness, fitness, p, \
                                     self._evals - self._last_move, size, position,\
                                     new_pos)
            print(s, file=sys.stderr)

        if np.random.random() < p:
            # protect against the kind of fruitless cycling that can occur if
            # two configurations have the same fitness
            if cur_fit != fitness:
                self._last_move = self._evals

            # update matrix
            self.current_fitness = fitness
            self.current_matrix = new_matrix
            self.current_order = self.current_order[new_order]
            #print self.current_order
            #print self.current_matrix[0]

            if self.current_fitness > self.best_fitness:
                self._last_best = self._evals
                self.best_fitness = self.current_fitness
                self.best_order = self.current_order
                self.best_matrix = self.current_matrix

                if self.verbose:
                    print('new_best: {bf}, {bo}'.format(
                        bf=self.best_fitness, bo=self.best_order))

        return fitness

    def fit_boxes(self):
        """not yet"""

        self.boxes = [[n] for n in range(self.matrix.size)]

        while 1:
            pass

    def _propose_box_move(self):

        box_pos = random.choice(len(self.boxes))
        candidate_box = self.boxes[box_pos]

        candidate = list(self.boxes)
        if random.random() < 0.5:
            # join
            if random.random() < 0.5:
                # join with the one to the left
                if box_pos == 0:
                    return
                else:
                    candidate[box_pos].extend(candidate[box_pos + 1])
                    candidate.pop(box_pos + 1)
                    return candidate
            else:
                # join with the one to the right
                if box_pos == len(self.boxes) - 1:
                    return
                else:
                    candidate[box_pos].extend(candidate[box_pos - 1])
                    candidate.pop(box_pos - 1)
                    return candidate

        else:
            # split
            if len(self.boxes[box_pos]) == 1:
                return
            else:
                split_pos = random.randrange(len(candidate_box) - 1)
                candidate.insert(box_pos, candidate_box[split_pos:])
                candidate_box = candidate_box[:split_pos]
                return candidate


def generate_test_data(size=64):
    """generate some data for testing the algorithm"""

    tiny_size = size / 16
    quad_size = size / 4
    half_size = size / 2
    a = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == j:
                a[i, j] = 1.
            elif i > j:
                a[i, j] = a[j, i]
            else:
                if i / tiny_size == j / tiny_size:
                    a[i, j] = np.random.normal(0.6, 0.05)
                elif i / quad_size == j / quad_size:
                    a[i, j] = np.random.normal(0.5, 0.05)
                elif i / half_size == j / half_size:
                    a[i, j] = np.random.normal(0.4, 0.05)
                else:
                    a[i, j] = np.random.normal(0.3, 0.05)

    # matrix = matrix(data=a, color_scheme='spectral', domain=(.1, .8),)
    # matrix.write_file('test_key.agr')
    df = pd.DataFrame(a)

    print('test key generated')
    df.to_csv('test_key.csv')
    return a


if __name__ == '__main__':
    data = generate_test_data()

    #file_name = sys.argv[1]
    #data = [map(float, line.strip().split()) \
    #        for line in open(file_name)]

    # with open('result.txt','w') as result:
    #     for n in order:
    #         print(n, file=result)
