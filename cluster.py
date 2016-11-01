from super_sort import SortingAlgorithm
import numpy as np
import random
from collections import abc

# box is a list + fitness...
class Box(list):
    def __init__(self, *args, **kwds):
        list.__init__(self, *args, **kwds)
        self.fitness = None

class BoxList(abc.MutableSequence):

    def __init__(self, boxes):
        self.boxes = boxes

    def __getitem__(self, key):
        return self.boxes[key]

    def __setitem__(self, key, value):
        self.boxes[key] = value

    def __delitem__(self, key):
        del self.boxes[key]

    def __len__(self):
        return len(self.boxes)

    def insert(self, key, value):
        self.boxes[key] = value


# potential functions
# box join
# box split
# box iter with front:back indicies


def bic(unique_points, likelihood, dof):
    return unique_points * np.log(likelihood) + (2 * dof + 1
                                                 ) * np.log(unique_points)

# TODO: recursive BoxClustering Algorithm
def hierarchical_box_clustering(self, branch=None):
    if branch is None:
        # ugly---is there a better way?
        self.hierarchy = Box()
        for i in range(self.matrix_size):
            self.hierarchy.append(i)
        branch = self.hierarchy
    matrix = self.current_matrix[branch, :][:, branch]
    branch.fitness = self._evaluate_box_fitness(
        boxes=[len(matrix)], matrix=matrix)
    boxes, fitness = self.fit_boxes(matrix=matrix, )
    boxes = [i + branch[0] for i in boxes]
    hierarchy = Box()
    for i, n in enumerate(boxes):
        if i == 0:
            b = Box()
            b.extend(range(branch[0], n))
            hierarchy.append(b)
        else:
            b = Box()
            b.extend(range(boxes[i - 1], n))
            hierarchy.append(b)
    hierarchy.fitness = fitness
    if len(hierarchy) == 1:
        return hierarchy
    for i, branch in enumerate(hierarchy):
        if len(branch) == 1:
            continue
        proposed = self.hierarchical_box_clustering(branch=branch)
        # TODO verify this mess...
        N = (len(branch)**2 + len(branch)) / 2.
        bic_partitioned = bic(N, proposed.fitness, len(proposed))
        bic_union = bic(N, branch.fitness, 1)
        print(branch)
        print(proposed)
        print(bic_partitioned, bic_union)
        print()
        if bic_partitioned < bic_union:
            hierarchy[i] = proposed[:]
    return hierarchy


class BoxClustering(SortingAlgorithm):

    def __init__(self, *args, **kwds):
        SortingAlgorithm.__init__(self, *args, **kwds)
        self.current_matrix = self.orig
        self.current_fitness = self.test_fitness(self.current_matrix)

    def fit_boxes(self, matrix=None):
        if matrix is None:
            matrix = self.current_matrix
        self.sub_graph = matrix
        self.box_temperature = 0.01
        self._box_t_since_last_move = 0
        # self._box_t_last_productive = 0
        self._box_evals = 0    # count turns
        # self.boxes is a list of the right-boundaries of the boxes
        self.boxes = [n + 1 for n in range(len(matrix))]
        # self.boxes = [len(matrix)]
        self.best_boxes = self.boxes[:]
        self.box_current_fitness = self._evaluate_box_fitness(self.boxes, matrix)
        self.box_best_fitness = self.box_current_fitness
        counter = 0  # why counter?
        while 1:
            if not (counter % len(matrix)):
                self.box_temperature *= 0.9
            self.box_turn()
            if self._box_t_since_last_move > len(matrix):
                break
            counter += 1
        return self.best_boxes, self.box_best_fitness

    def _evaluate_box_fitness(self, boxes=None, matrix=None):
        """least squares
        sum of squared error from mean within every box
        + the squared error from mean of every cell outside of a box
        """
        if matrix is None:
            matrix = self.current_matrix
        if boxes is None:
            boxes = self.boxes
        fitness = 0.
        # evaluate the least-squares fitness for each box
        non_box_nodes = np.ones((len(matrix), len(matrix)), dtype=bool)
        for i, box in enumerate(boxes):
            if i == 0:
                begin = 0
            else:
                begin = boxes[i - 1]
            end = box
            box_nodes = matrix[begin:end, begin:end]
            non_box_nodes[begin:end, begin:end] = False
            m = box_nodes.mean()
            sq = (box_nodes - m)**2
            fitness += sq.sum()
        # now do it for the non-box nodes
        non_box = matrix[non_box_nodes]
        m = non_box.mean()
        sq = (non_box - m)**2
        fitness += sq.sum()
        return fitness

    def box_turn(self):
        """

        1. propose move
        2. calculate fitness
        3. calculate probability of accepting move (based on fitness differential)
        4. accept move or don't
        ....
        travel around while always keeping track of best location passed through

        self._box_t_since_last_move increments in three postions
        a) box_turn() starts
        b) p is accepted
        b2) again if b) happens and current fitness is different than regular fitness
        ....
        why? three times?
        """
        # set counters
        self._box_evals += 1
        self._box_t_since_last_move += 1
        # propose move
        candidate = self._propose_box_move()
        if not candidate:
            return

        # reset parameters
        cur_fit = self.box_current_fitness
        fitness = self._evaluate_box_fitness(candidate, matrix=self.sub_graph)
        self.box_best_fitness = self.current_fitness
        self.best_boxes = self.boxes

        # probability of accepting move
        p = np.exp((cur_fit - fitness) / cur_fit / self.box_temperature)
        if np.random.random() < p:
            if cur_fit != fitness:
                self._box_t_since_last_move += 1
            # update
            self.box_current_fitness = fitness
            self.boxes = candidate
            # keep track of best fitness
            if fitness < self.box_best_fitness:
                self.box_best_fitness = fitness
                self.best_boxes = candidate
                self._box_t_since_last_move = 0
                # print(self.box_best_fitness, self.box_temperature, self.best_boxes)
            self._box_t_since_last_move += 1

    def _propose_box_move(self):
        """
        1. pick a random position in box list
        2. randomly pick if doing a split or join on position

        if split -
        ignore
                if candidate[box_pos] == 1:
        or
                if (candidate[box_pos] - candidate[box_pos - 1]) == 1:
        ... why?

        if join - pick left or right.
        join current box with that box.

        """
        box_pos = random.randrange(len(self.boxes))
        # XXX debug
        # candidate_box = self.boxes[box_pos]
        # old_candidate = self.boxes[:]
        candidate = list(self.boxes)
        # split
        if random.random() < 0.5:
            if box_pos == 0:
                if candidate[box_pos] == 1:
                    return
                cut_location = random.randrange(1, candidate[box_pos])
            else:
                if (candidate[box_pos] - candidate[box_pos - 1]) == 1:
                    return
                cut_location = random.randrange(candidate[box_pos - 1],
                                                candidate[box_pos])
            candidate.insert(box_pos, cut_location)

        # join
        else:
            if random.random() < 0.5:
                # join with the one to the left
                if box_pos == 0:
                    return
                else:
                    if box_pos == 1:
                        lower = 0
                        target = candidate[box_pos - 1]
                        d = np.inf
                        while d > (target - lower):
                            d = int(np.ceil((np.random.pareto(0.5))))
                        cut_location = target - d
                        # cut_location = random.randrange(candidate[box_pos-1])
                        if cut_location == 0:
                            candidate.pop(box_pos - 1)
                        else:
                            candidate[box_pos - 1] = cut_location
                    else:
                        lower = candidate[box_pos - 2]
                        target = candidate[box_pos - 1]
                        d = np.inf
                        while d > (target - lower):
                            d = int(np.ceil((np.random.pareto(0.5))))
                        cut_location = target - d
                        # cut_location = random.randrange(candidate[box_pos-2], candidate[box_pos-1])
                        if cut_location == candidate[box_pos - 2]:
                            candidate.pop(box_pos - 1)
                        else:
                            candidate[box_pos - 1] = cut_location
                    return candidate
            else:
                # join with the one to the right
                if box_pos == len(self.boxes) - 1:
                    return
                else:
                    target = candidate[box_pos]
                    upper = candidate[box_pos + 1]
                    d = np.inf
                    while d > (upper - target):
                        d = int(np.ceil((np.random.pareto(0.5))))
                    cut_location = target + d
# cut_location = random.randrange(candidate[box_pos]+1, candidate[box_pos+1]+1)
                    if cut_location == candidate[box_pos + 1]:
                        candidate.pop(box_pos)
                    else:
                        candidate[box_pos] = cut_location
                    return candidate
