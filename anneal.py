import numpy as np
import pandas as pd
from itertools import count
from .boxsort import SortingAlgorithm
from collections import namedtuple
from collections import abc


class BaseAnnealer(object):

    def _probability_to_accept(self, improvement):
        return np.exp(improvement / self.temp)

    def accept_move(self, cur_fit, new_fit):
        new_fit_larger = new_fit > cur_fit
        delta = new_fit - cur_fit

        if self.maximize:
            if new_fit_larger:
                return True
        else:
            if not new_fit_larger:
                return True
            delta = -delta

        if np.abs(cur_fit) < 0.0001:
            cur_fit == 0.0001
        improvement = delta / cur_fit
        p = self._probability_to_accept(improvement)
        return np.random.random() < p

    def decrease_temp(self, fraction_moved=0.2):
        if fraction_moved > 0.1:
            self.temp *= self.cooling_factor**100
        else:
            c_f = fraction_moved * 1000.
            self.temp *= self.cooling_factor**int(np.ceil(c_f))

    def _temp_block_finished(self, i, block_size):
        return not (i % block_size)


class Annealer(BaseAnnealer):

    # __metaclass__ = abc.ABCMeta
    maximize = True
    Trace = namedtuple('trace', ('evals',
                                 'last_move',
                                 'last_best',
                                 'moves_this_temp',
                                 'move_accepted',
                                 'temp',
                                 'current_fit',
                                 'new_fit',
                                 'best_fit'))

    # @abc.abstractmethod
    def propose_move(self):
        # propose move
        pass

    # @abc.abstractmethod
    def evaluate_fitness(self, candidate):
        candidate.calculate_fitness()

    def __init__(self, obj):
        self.current = obj

    def turn(self, i):
        candidate = self.propose_move()
        new_fit = candidate.fitness
        cur_fit = self.current.fitness

        move_accepted = False
        self._update(candidate)
        if self.accept_move(cur_fit, new_fit):
            self.current = candidate
            move_accepted = True
            self._since_last_move = 0
        return self.make_trace(i, move_accepted, candidate)

    def make_trace(self, i, move_accepted, candidate):
        return self.Trace(i,
                          self._since_last_move,
                          self._since_last_best,
                          move_accepted,
                          self._moves_this_temp,
                          self.temp,
                          self.current.fitness,
                          candidate.fitness,
                          self.best.fitness)

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
            state = self.turn(i)
            # break if done
            if self._break_condition(state):
                break
            # save states if appropriate
            if self.save_history:
                self.history.append(state)
            # increment temperature
            if self._temp_block_finished(i, block_size):
                state = self.decrease_temp(state)
        return self.obj

    def _update(self, candidate):

        self._since_last_move += 1
        self._since_last_best += 1
        if candidate < self.best:
            self.best = candidate.copy()
            self._since_last_best = 0

    def _initialize_state(self, matrix, boxes=None):
        pass
