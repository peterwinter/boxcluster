#!/usr/bin/env python
"""structures for performing a couple of different clustering algorithms. most
notably, performs matrix sorting as described in marta's pnas, using simulated
annealing"""

import sys
import numpy as np
import random
import math
import pandas as pd
from scipy.stats import expon
from scipy.cluster import hierarchy


class Box(list):
    def __init__(self, *args, **kwds):
        list.__init__(self, *args, **kwds)
        self.fitness = None


def bic(unique_points, likelihood, dof):
    return unique_points * np.log(likelihood) + (2*dof+1)* np.log(unique_points)


class ClusterMatrix(np.ndarray):


    def __new__(cls, data, *args, **kwargs):

        a = np.array(data).view(cls)

        a.hierarchical_clustering = HierarchicalClustering(a)
        a.deep_sort = SASort(a)

        return a


class SortingAlgorithm(object):
    """base class for sorting algorithms. this establishes the weight matrix,
and contains methods for moving things around in the matrix, but none of the
meat-and-potatoes of the algorithm."""


    def __init__(self, original_matrix):

        self.matrix_size = len(original_matrix)
        self.orig = original_matrix

        # build a weight matrix that falls away from the diagonal this will be
        # useful later when
        self.weight_matrix = np.ones(self.orig.shape)
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                self.weight_matrix[i,j] -= (abs(i-j)/float(self.matrix_size))


    def reorder(self, order, matrix=None):
        """put the matrix in a new order"""
        if matrix == None:
            matrix = self.current_matrix
        return matrix[order,:][:,order]


    def smart_reorder(self, order, matrix_range, matrix=None):
        """reorder the matrix, but do it a faster way. most moves will only
affect a small fraction of the matrix, so there's no reason to move most of the
elements"""
        if matrix == None:
            matrix = self.current_matrix.copy()
        else:
            matrix = matrix.copy()
        start = matrix_range[0]
        stop = matrix_range[-1]+1
        sub_matrix = self.current_matrix[order,:]

        matrix[start:stop,:] = sub_matrix


        sub_matrix = matrix[:, order]
        matrix[:, start:stop] = sub_matrix

        return matrix


    def test_fitness(self, array=None):
        """returns the fitness of the matrix, which is the product of the
current adjacency matrix and the weight matrix"""
        if array == None:
            array = self.current_matrix

        return (array * self.weight_matrix).sum()


class HierarchicalClustering(SortingAlgorithm):
    """use scipy.cluster.hierarchy to sort the matrix"""

    def __call__(self):

        self.order = hierarchy.leaves_list(hierarchy.linkage(self.orig))
        self.result = self.reorder(self.order, self.orig)
        return self.result, self.order



class SASort(SortingAlgorithm):

    def __init__(self, *args, **kwds):

        SortingAlgorithm.__init__(self, *args, **kwds)


        # set the counters and empty the files
        self._evals = 0
        self._last_move = 0
        self._last_best = 0
        # keep track of these in case something gets interrupted
        open('sort_save.txt','w').close()
        open('fit_save.txt','w').close()

        self.current_matrix = self.orig
        self.current_order = np.arange(len(self.orig))

        # log initial configuration
        self.current_fitness = self.test_fitness(self.current_matrix)
        with open('initial_fitness.txt', 'w') as f:
            print(self.current_fitness, file=f)
        # shuffle the matrix and initialize tracker variables


    def initial_shuffle(self):
        order = self.current_order
        random.shuffle(order)
        self.current_order = np.array(order)

        self.current_matrix = self.reorder(self.current_order)


    def __call__(self, cooling_factor=0.999, temperature=0.001,
                        finishing_criterion=1, verbosity=False):
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
        is frozen. practically, there is a finite probability that we're not in a
        global optimum. as the system cools, the probability of finding a
        better solution decreases. this parameter defines how long (how many
        cooling steps) we're willing to spin the wheels before accepting a
        solution.
    verbose (default: False)
        how much shit to print out


        """
        self.initial_shuffle()
        self.current_fitness = self.test_fitness(self.current_matrix)

        self.best_matrix = self.current_matrix
        self.best_order = self.current_order
        self.best_fitness = self.test_fitness(self.current_matrix)



        self.verbosity = verbosity
        self.temperature = temperature
        self.cooling_factor = cooling_factor
        self.finishing_criterion = finishing_criterion
        self._delta = 0.

        steps = int(self.matrix_size*(100 + self.matrix_size)/100)
        self.steps = steps


        # keep track of the time since the last move
        last_move_time = 0

        with open('best_order.txt', 'w') as h:
            # each loop is a temperature
            while 1:

                # start a new temperature
                self._moves_this_temp = 0
                for n in range(steps):

                    # keep track of the number of productive moves at this
                    # temperature so we can adaptively change the temp

                    self.turn()
                    last_move_time += self._evals - self._last_move

                for i in self.best_order:
                    print(i, file=h)

                fraction_moved = float(self._moves_this_temp) / float(steps)
                since_last_best = self._evals - self._last_best

                print(self.temperature, fraction_moved, self.best_fitness,\
                np.ceil(1000. * fraction_moved), self._evals - self._last_move,\
                steps * self.finishing_criterion)

                self._save()
                # if it has been a couple of cooling cycles since we last improved,
                # then return. this should be long enough to search the local
                # landscape and find the local maximum.
                if self._evals - self._last_move > steps * self.finishing_criterion:
                    return self.best_matrix, self.best_order

                if fraction_moved >0.1:
                    self.temperature *= self.cooling_factor**100
                else:
                    c_f= fraction_moved * 1000.
                    self.temperature *= self.cooling_factor**int(np.ceil(c_f))

                #if since_last_best > steps * self.finishing_criterion:
                #    self.temperature *= self.cooling_factor ** 5


    def _save(self):
        """cleanup that occurs at the end of a cooling step"""

        sys.stdout.flush()

        with open('fit_save.txt', 'a') as fit_handle:
            print((self.current_fitness, self.best_fitness), file=fit_handle)


    def turn(self):

        self._evals += 1


        # roll the dice for the size of the slice
        size = np.inf
        while size >= self.matrix_size:
            size = np.random.pareto(0.2)
            #size = expon.rvs(0., self.matrix_size/5.)*(1. - self._delta)
            if size < 1:
                size = 1 # for some reason, this was returning negative values
            size = int(math.ceil(size))

        # roll the dice for the location---the starting position of the slice
        position = random.randrange(0, self.matrix_size-size)

        # move it a distance
        # XXX this should probably depend on the temp!
        while 1:
            #new_pos = int(math.ceil(np.random.pareto(0.7)))
            #new_pos = int(math.ceil(expon.rvs(0., self.matrix_size/5.)))
            new_pos = np.random.pareto(0.2)
            new_pos = int(math.ceil(new_pos))

            # random sign
            if np.random.random() < 0.5:
                new_pos = -new_pos

            if 0 <= (position + new_pos) <= (self.matrix_size - size ):
                break

        order = list(range(self.matrix_size))

        # propose a new order
        if new_pos > 0 :
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
            p =  np.exp( (fitness - cur_fit) / cur_fit / self.temperature )

        if self.verbosity==2:
            a = (self._evals, self.temperature,
                 self.current_fitness, fitness, p,
                 self._evals - self._last_move,
                 size, position, new_pos)
            s = '%6d %.5e %.12e %.8e %.3e\t%5d %5d %5d %+5d'%a
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
            #print self.current_order
            #print self.current_matrix[0]

            if self.current_fitness > self.best_fitness:
                if self.verbosity==1:
                    a = (self._evals, self.temperature, \
                         self.current_fitness, fitness, p, \
                         self._evals - self._last_move, size, position,\
                         new_pos)
                    print('%6d %.5e %.12e %.8e %.3e\t%5d %5d %5d %+5d'%a)


                self._last_best = self._evals
                self.best_fitness = self.current_fitness
                self.best_order = self.current_order
                self.best_matrix = self.current_matrix

        # how close we are to quitting
        # print(self._evals)
        # print(self._last_move)
        # print(self.steps)
        # print(self.finishing_criterion)

        self._delta = (self._evals - self._last_move) / self.steps / self.finishing_criterion
        return fitness

    def hierarchical_box_clustering(self, branch=None):

        if branch==None:
            # ugly---is there a better way?
            self.hierarchy = Box()
            for i in range(self.matrix_size):
                self.hierarchy.append(i)
            branch=self.hierarchy


        matrix=self.current_matrix[branch,:][:,branch]
        branch.fitness = self._evaluate_box_fitness(boxes=[len(matrix)],
                                                        matrix=matrix)


        boxes, fitness = self.fit_boxes(matrix=matrix,)

        boxes = [i+branch[0] for i in boxes]


        hierarchy = Box()
        for i, n in enumerate(boxes):
            if i == 0:
                b = Box()
                b.extend(range(branch[0],n))
                hierarchy.append(b)
            else:
                b = Box()
                b.extend(range(boxes[i-1],n))
                hierarchy.append(b)

        hierarchy.fitness = fitness
        if len(hierarchy) == 1:
            return hierarchy


        for i,branch in enumerate(hierarchy):
            if len(branch) == 1:
                    continue
            proposed = self.hierarchical_box_clustering(branch=branch)

            # TODO verify this mess...
            N = (len(branch)**2 + len(branch))/2.
            bic_partitioned = bic(N, proposed.fitness, len(proposed))
            bic_union = bic(N, branch.fitness, 1)

            print(branch)
            print(proposed)
            print(bic_partitioned, bic_union)
            print()
            if bic_partitioned < bic_union:
                hierarchy[i] = proposed[:]

        return hierarchy


    def fit_boxes(self, matrix=None):

        if matrix == None:
            matrix = self.current_matrix

        self.sub_graph = matrix

        self.box_temperature=0.01
        self._box_t_since_last_move = 0
        self._box_t_last_productive = 0
        self._box_evals = 0

        # self.boxes is a list of the right-boundaries of the boxes

        self.boxes = [n+1 for n in range(len(matrix))]
        #self.boxes = [len(matrix)]
        self.best_boxes = self.boxes[:]

        self.box_current_fitness = self._evaluate_box_fitness(self.boxes, matrix)
        self.box_best_fitness=self.box_current_fitness


        counter = 0
        while 1:
            if not (counter % len(matrix)):
                self.box_temperature *= 0.9
            self.box_turn()

            if self._box_t_since_last_move > len(matrix):
                break
            counter +=1
        return self.best_boxes, self.box_best_fitness


    def _evaluate_box_fitness(self, boxes=None, matrix=None):
        """least squares"""

        if matrix == None:
            matrix = self.current_matrix
        if boxes == None:
            boxes = self.boxes

        fitness = 0.
        # evaluate the least-squares fitness for each box

        non_box_nodes = np.ones((len(matrix),len(matrix)))
        for i, box in enumerate(boxes):
            if i == 0:
                begin = 0
            else:
                begin = boxes[i-1]
            end = box


            box_nodes = matrix[begin:end, begin:end]
            non_box_nodes[begin:end, begin:end] = 0

            m = box_nodes.mean()
            sq = (box_nodes - m) ** 2
            fitness += sq.sum()


        # now do it for the non-box nodes
        non_box = non_box_nodes * matrix
        m = non_box.mean()
        sq = (non_box - m) ** 2
        fitness += sq.sum()

        return fitness

    def box_turn(self):
        self._box_evals += 1
        self._box_t_since_last_move +=1

        candidate = self._propose_box_move()

        if not candidate:
            return

        cur_fit = self.box_current_fitness
        fitness = self._evaluate_box_fitness(candidate, matrix=self.sub_graph)
        self.box_best_fitness = self.current_fitness
        self.best_boxes = self.boxes



        p =  np.exp( ( cur_fit - fitness) / cur_fit / self.box_temperature )
        if np.random.random() < p:
            if cur_fit != fitness:

                self._box_t_since_last_move += 1

            # update
            self.box_current_fitness = fitness
            self.boxes = candidate

            if self.box_current_fitness < self.box_best_fitness:
                self.box_best_fitness = self.box_current_fitness
                self.best_boxes = self.boxes
                self._box_t_since_last_move = 0
                print(self.box_best_fitness, self.box_temperature, self.best_boxes)

            self._box_t_since_last_move +=1



    def _propose_box_move(self):
        box_pos = random.randrange(len(self.boxes))

        # XXX debug
        candidate_box = self.boxes[box_pos]
        old_candidate = self.boxes[:]

        candidate = list(self.boxes)

        # split
        if random.random() < 0.5:


            if box_pos == 0:
                if candidate[box_pos] == 1:
                    return
                cut_location = random.randrange(1, candidate[box_pos])
            else:
                if (candidate[box_pos] - candidate[box_pos-1]) == 1:
                    return
                cut_location = random.randrange(candidate[box_pos-1], candidate[box_pos])

            candidate.insert(box_pos, cut_location)

        # join
        else:
            if random.random() < 0.5:

                # join with the one to the left
                if box_pos == 0 :
                    return
                else:
                    if box_pos == 1:

                        lower = 0
                        target = candidate[box_pos-1]
                        d = np.inf
                        while d > (target - lower):
                            d = int(np.ceil((np.random.pareto(0.5))))

                        cut_location = target - d
                        #cut_location = random.randrange(candidate[box_pos-1])
                        if cut_location == 0:
                            candidate.pop(box_pos-1)
                        else:
                            candidate[box_pos-1] = cut_location

                    else:

                        lower = candidate[box_pos - 2]
                        target = candidate[box_pos - 1]
                        d = np.inf
                        while d > (target - lower):
                            d = int(np.ceil((np.random.pareto(0.5))))

                        cut_location = target - d
                        #cut_location = random.randrange(candidate[box_pos-2], candidate[box_pos-1])



                        if cut_location == candidate[box_pos-2]:
                            candidate.pop(box_pos-1)

                        else:
                            candidate[box_pos-1] = cut_location


                    return candidate
            else:

                # join with the one to the right
                if box_pos == len(self.boxes) - 1:
                    return

                else:

                    target = candidate[box_pos]
                    upper = candidate[box_pos+1 ]
                    d = np.inf
                    while d > (upper - target):
                        d = int(np.ceil((np.random.pareto(0.5))))

                    cut_location = target + d
                    #cut_location = random.randrange(candidate[box_pos]+1, candidate[box_pos+1]+1)
                    if cut_location == candidate[box_pos + 1]:
                        candidate.pop(box_pos)
                    else:
                        candidate[box_pos] = cut_location

                    return candidate



def generate_test_data(size=64):
    """generate some data for testing the algorithm"""

    tiny_size = size/16
    quad_size = size/4
    half_size = size/2
    a = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == j:
                a[i,j] =  1.
            elif i > j:
                a[i,j] = a[j,i]
            else:
                if int(i/tiny_size) == int(j/tiny_size):
                    a[i,j] = np.random.normal(0.6, 0.05)
                elif int(i/quad_size) == int(j/quad_size):
                    a[i,j] = np.random.normal(0.5, 0.05)
                elif int(i/half_size) == int(j/half_size):
                    a[i,j] = np.random.normal(0.4, 0.05)
                else:
                    a[i,j] = np.random.normal(0.3, 0.05)

    # matrix = matrix(data=a,
    #                  color_scheme='spectral',
    #                  domain=(.1, .8),
    #                                  )

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

    c = ClusterMatrix(data)

    (data, order) = c.deep_sort(cooling_factor=0.96, verbose=True,
                                finishing_criterion=3)

    with open('result.txt','w') as result:
        for n in order:
            print(n, file=result)
