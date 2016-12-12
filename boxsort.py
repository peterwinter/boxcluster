#!/usr/bin/env python
"""structures for performing a couple of different clustering algorithms. most
notably, performs matrix sorting as described in marta's pnas, using simulated
annealing"""

from itertools import count

from .anneal import Annealer
from .boxorder import OrderedArray


class BoxSort(Annealer):

    maximize = True

    def __init__(self, matrix):
        self.matrix_size = len(matrix)
        self.matrix = matrix

        self._since_last_move = 0
        self._moves_this_temp = 0

        cooling_factor = 0.999
        temperature = 0.001
        finishing_criterion = 1

        self.cooling_factor = cooling_factor
        self.temp = temperature
        self.finishing_criterion = finishing_criterion

    def propose_move(self):
        candidate = self.current.copy()
        candidate.propose_move()
        return candidate

    def _initialize_state(self, order=None):
        self._since_last_move = 0
        self._moves_this_temp = 0
        current = OrderedArray(self.matrix, order=order)
        self.current = current

    def __call__(self,
                 cooling_factor=0.999,
                 temperature=0.001,
                 finishing_criterion=1,
                 order=None,
                 save_history=False):

        # save input parameters
        self.cooling_factor = cooling_factor
        self.temp = temperature
        self.finishing_criterion = finishing_criterion
        self.history = []

        # determine temperature block size
        block_size = int(self.matrix_size * (100 + self.matrix_size) / 100)
        self.finishing_threshold = self.finishing_criterion * block_size

        self._initialize_state(order=order)

        for i in count():
            trace = self.turn(i, only_count_best=True)
            # print(trace)
            # break if done
            if self._break_condition():
                break
            # save states if appropriate
            if save_history:
                self.history.append(trace)
            # increment temperature
            if self._temp_block_finished(i, block_size):
                fraction_moved = float(self._moves_this_temp) / block_size
                self.decrease_temp(fraction_moved)
                self._moves_this_temp = 0

            if i > 10000:
                break
        return self.current

    def _break_condition(self):
        if self._since_last_move > self.finishing_threshold:
            return True
        return False

    # def decrease_temp(self):
    #     # TODO: Actually calculate fraction moved...
    #     fraction_moved = 0.2
    #     if fraction_moved > 0.1:
    #         self.temp *= self.cooling_factor**100
    #     else:
    #         c_f = fraction_moved * 1000.
    #         self.temp *= self.cooling_factor**int(np.ceil(c_f))

    # unchanged from anneal...
    def _temp_block_finished(self, i, block_size):
        return not (i % block_size)
