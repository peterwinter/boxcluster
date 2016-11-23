import numpy as np
import pandas as pd
from itertools import count
from .boxsort import SortingAlgorithm
from collections import namedtuple
from collections import abc


class BaseAnnealer(object):


    def _probability_to_accept(self, improvement):
        return np.exp(improvement / self.box_temperature)

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
            self.temperature *= self.cooling_factor**100
        else:
            c_f = fraction_moved * 1000.
            self.temperature *= self.cooling_factor**int(np.ceil(c_f))

    def _temp_block_finished(self, i, block_size):
        return not (i % block_size)

    def _break_condition(self, state):
        pass

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

    def turn(self, trace):

        state.evals += 1
        candidate = self.propose_move()
        new_fit = self.candidate.fitness
        cur_fit = self.current.fitness
        if self.move_accepted(cur_fit, new_fit):
            self.current = candidate
            self._update(state)
        return trace

    def __call__(self,
                 cooling_factor=0.999,
                 temperature=0.001,
                 finishing_criterion=1,
                 save_history=False):
        # save input parameters
        self.cooling_factor = cooling_factor
        self.temperature = temperature
        self.finishing_criterion = finishing_criterion
        self.history = []

        # run solver
        block_size = 100
        state = self._initialize_state()
        for i in count():
            state = self.turn(state)
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
        if candidate < self.best:
            self.best = candidate.copy()
            self._box_t_since_last_move = 0

    def _initialize_state(self, matrix, boxes=None):
        pass
