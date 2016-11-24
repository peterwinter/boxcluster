import numpy as np
import pandas as pd
from itertools import count
from .boxlist import BoxList
from .boxsort import SortingAlgorithm
from .anneal import Annealer


class BoxCut(SortingAlgorithm):
    def __init__(self, original_matrix):
        super().__init__(original_matrix)
        self.current_matrix = self.orig

    def box_turn(self):
        """
        1. propose move
        2. calculate fitness
        3. calculate probability of accepting move
        (based on fitness differential)
        4. accept move or don't
        """
        # update counters
        self._box_evals += 1
        self._box_t_since_last_move += 1
        candidate = self.propose_box_move(boxes=self.boxes)
        if not candidate:
            return
        # reset parameters
        self._update_if_best(candidate)
        self.best_boxes = self.boxes
        move_accepted = False
        if self.move_is_accepted(
                cur_fit=self.boxes.fitness, new_fit=candidate.fitness):
            # why this extra counter? try removing
            # if improvement_ratio != 0:
            #     self._box_t_since_last_move += 1
            self.boxes = candidate
            self._update_if_best(candidate)
            move_accepted = True

        record = (self._box_evals, self._box_t_since_last_move, move_accepted,
                  self.box_temperature, self.boxes.fitness, candidate.fitness,
                  self.best_boxes.fitness, len(self.boxes), len(candidate),
                  len(self.best_boxes))
        return record

    def _probability_to_accept(self, improvement):
        """
        p = 1 when no difference in fitness
        p > 1 when improvement (ie. always accept)
        p gets small very quickly with negative improvement
        """
        return np.exp(improvement / self.box_temperature)

    def move_is_accepted(self, cur_fit, new_fit):
        if new_fit < cur_fit:
            return True
        else:
            improvement = (cur_fit - new_fit) / cur_fit
            p = self._probability_to_accept(improvement)
            return np.random.random() < p

    def propose_box_move(self, boxes):
        # propose move
        candidate = boxes.propose_move()
        if not candidate:
            return
        # calculate fitness
        self._evaluate_box_fitness(candidate, matrix=self.sub_graph)
        return candidate

    def debug(self, matrix):
        it = self._iter_solve(matrix=matrix, debug=True)
        head = list(it.__next__())
        df = pd.DataFrame(list(it), columns=head).set_index('evals')
        return df

    def _update_if_best(self, candidate):
        if candidate < self.best_boxes:
            self.best_boxes = candidate
            self._box_t_since_last_move = 0

    def _initialize_search(self, matrix, boxes=None):
        self.sub_graph = matrix
        self.box_temperature = 0.05
        self._box_t_since_last_move = 0
        self._box_evals = 0  # count turns
        self.boxes = boxes
        if boxes is None:
            self.boxes = BoxList([n + 1 for n in range(len(matrix))])
        self._evaluate_box_fitness(self.boxes, matrix)
        self.best_boxes = self.boxes

    def fit_boxes(self, matrix=None, boxes=None):
        if matrix is None:
            matrix = self.current_matrix
        for i in self._iter_solve(matrix=matrix, boxes=boxes):
            pass
        return self.best_boxes, self.best_boxes.fitness

    def _iter_solve(self, matrix, debug=False, boxes=None):
        self._initialize_search(matrix=matrix, boxes=boxes)
        if debug:
            head = ('evals', 't_since_last_move', 'move_accepted', 'temp',
                    'current_fit', 'new_fit', 'best_fit', 'current_len',
                    'new_len', 'best_len')
            yield head

        n = len(matrix)
        lower_limit = 100
        if n < lower_limit:
            n = lower_limit

        for i in count():
            # every n iterations, lower temp
            if not (i % n):
                self.box_temperature *= 0.9
            # Key Line!
            debug_feed = self.box_turn()
            if self._box_t_since_last_move > n:
                break
            if debug and debug_feed is not None:
                yield debug_feed

    def _evaluate_box_fitness(self, boxes=None, matrix=None):
        """least squares
        sum of squared error from mean within every box
        + the squared error from mean of every cell outside of a box
        """
        if matrix is None:
            matrix = self.current_matrix
        if boxes is None:
            boxes = self.boxes
        boxes.calculate_fitness(matrix)


class BoxCut2(Annealer):

    maximize = False

    def __init__(self, matrix):
        self.matrix_size = len(matrix)
        self.matrix = matrix
        self._since_last_move = 0
        self._since_last_best = 0
        self._moves_this_temp = 0

        cooling_factor = 0.999
        temperature = 0.001
        finishing_criterion = 1
        self.cooling_factor = cooling_factor
        self.temp = temperature
        self.finishing_criterion = finishing_criterion

    def propose_move(self):
        # propose move
        # print(self.current)
        candidate = None
        while candidate is None:
            candidate = self.current.propose_move()
        # print(candidate)
        # propose move
        # calculate fitness
        self.evaluate_fitness(candidate, matrix=self.matrix)
        return candidate

    def evaluate_fitness(self, candidate, **kwargs):
        candidate.calculate_fitness(**kwargs)

    def _initialize_state(self, boxes=None):
        self._since_last_move = 0
        self._since_last_best = 0
        self._moves_this_temp = 0
        matrix = self.matrix
        if boxes is None:
            boxes = BoxList([n + 1 for n in range(len(matrix))])
        self.evaluate_fitness(candidate=boxes, matrix=matrix)
        self.current = boxes
        self.best = boxes.copy()

    def __call__(self,
                 cooling_factor=0.999,
                 temperature=0.001,
                 finishing_criterion=1,
                 boxes=None,
                 save_history=False):

        # save input parameters
        self.cooling_factor = cooling_factor
        self.temp = temperature
        self.finishing_criterion = finishing_criterion
        self.history = []

        # run solver
        block_size = 100
        # create self.current, self.best
        self._initialize_state(boxes=boxes)

        for i in count():
            trace = self.turn(i)
            # print(trace)
            # break if done
            if self._break_condition():
                break
            # save states if appropriate
            if save_history:
                self.history.append(trace)
            # increment temperature
            if self._temp_block_finished(i, block_size):
                self.decrease_temp()

            if i > 1000:
                break

        return self.best

    def _break_condition(self):
        pass

        n = len(self.matrix)
        lower_limit = 100
        if n < lower_limit:
            n = lower_limit

        if self._since_last_move > n:
            return True
        return False

    def decrease_temp(self):
        # TODO: make depend on cooling factor
        self.temp *= 0.9
