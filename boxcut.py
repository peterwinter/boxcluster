import numpy as np
from itertools import count
from .boxlist import BoxList
from .anneal import Annealer

class BoxCut(Annealer):

    maximize = False

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
        self._moves_this_temp = 0
        matrix = self.matrix
        if boxes is None:
            boxes = BoxList([n + 1 for n in range(len(matrix))])
        self.evaluate_fitness(candidate=boxes, matrix=matrix)
        self.current = boxes

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
        block_size = self.matrix_size
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

        return self.current

    def _break_condition(self):
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
