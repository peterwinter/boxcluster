import numpy as np
import networkx as nx
from itertools import count
from .mod import Modules
from .anneal import Annealer


def largest_connected(G):
    return next(nx.connected_component_subgraphs(G))


class BaseModMax(Annealer):

    def graph_check(self, G):
        if G.__class__ == nx.classes.graph.Graph:
            n_components = nx.number_connected_components(G)
            if n_components == 1:
                return G
            elif n_components > 1:
                return largest_connected(G)
            else:
                return None
        else:
            return None

    def get_subgraphs(self, modules):
        '''For each module m in a given list of modules, produces the subgraph
        containing only the nodes in m.'''
        subgraphs = []
        # print(modules)
        for s in modules:
            # print(s)
            if s:
                subgraphs.append(self.G.subgraph(s))
        return subgraphs

    def calc_single_move_n(self, fac, num_nodes):
        if (fac * num_nodes**2 < 10):
            num_individual_moves = 10
        else:
            num_individual_moves = int(fac * num_nodes**2)
        return num_individual_moves

    def calc_group_move_n(self, fac, num_nodes):
        if (fac * num_nodes < 2):
            n_group_moves = 2
        else:
            n_group_moves = int(fac * num_nodes)
        return n_group_moves


class ModMax(BaseModMax):

    maximize = True

    def __init__(self, network):
        # self.matrix_size = len(matrix)
        # self.matrix = matrix
        self.G = self.graph_check(network)
        self.L = len(self.G)
        self._since_last_move = 0
        self._moves_this_temp = 0

        cooling_factor = 0.999
        temperature = 0.001
        finishing_criterion = 1

        self.cooling_factor = cooling_factor
        self.temp = temperature
        self.finishing_criterion = finishing_criterion

    def evaluate_fitness(self, candidate, **kwargs):
        candidate.calculate_fitness(**kwargs)

    def _initialize_state(self, modules=None, n_modules=2):
        self._since_last_move = 0
        self._moves_this_temp = 0
        if modules is None:
            modules = Modules.random_initialization(
                G=self.G, L=self.L, num_modules=n_modules)
        self.current = modules

    def __call__(self,
                 cooling_factor=0.999,
                 temperature=0.001,
                 finishing_criterion=1,
                 modules=None,
                 n_modules=2,
                 save_history=False):
        # save input parameters
        self.cooling_factor = cooling_factor
        self.temp = temperature
        self.finishing_criterion = finishing_criterion
        self.history = []
        # run solver
        # create self.current
        self._initialize_state(modules=modules, n_modules=n_modules)

        fac = 0.80
        num_nodes = self.L

        single_moves = self.calc_single_move_n(fac, num_nodes)
        group_moves = self.calc_group_move_n(fac, num_nodes)

        i = 0
        for j in count():
            for _ in range(single_moves):
                candidate = self.propose_single_move()
                trace = self.evaluate_move(candidate, i)
                if save_history:
                    self.history.append(trace)
                i += 1

            for _ in range(group_moves):
                candidate = self.propose_group_move()
                trace = self.evaluate_move(candidate, i)
                if save_history:
                    self.history.append(trace)
                i += 1

            if self._break_condition():
                break
            self.decrease_temp()
        return self.current

    def evaluate_move(self, candidate, i):
        new_fit = candidate.fitness
        cur_fit = self.current.fitness
        move_accepted = False
        self._update(candidate)
        if self.accept_move(cur_fit, new_fit):
            self.current = candidate
            move_accepted = True
            self._since_last_move = 0
        return self.make_trace(i, move_accepted, candidate)

    def propose_single_move(self):
        candidate = self.current.copy()
        candidate.move_random_single()
        self.evaluate_fitness(candidate, G=self.G, L=self.L)
        return candidate

    def propose_group_move(self):
        candidate = self.current.copy()
        move_type = np.random.choice(['split', 'merge'])
        # if one type of move doesn't work do the other type
        move_occured = False
        i = 0
        while move_occured is False:
            if move_type == 'merge':
                move_occured = candidate.move_random_merge()
                move_type = 'split'
            if move_type == 'split':
                move_occured = candidate.move_random_split()
                move_type = 'merge'
            # add this for safeguard if neither type works
            i += 1
            if i > 2:
                print('WTF! this should never happen')
        self.evaluate_fitness(candidate, G=self.G, L=self.L)
        return candidate

    def _break_condition(self):
        n = self.L * 2
        lower_limit = 100
        if n < lower_limit:
            n = lower_limit
        if self._since_last_move > n:
            return True
        return False

    def decrease_temp(self):
        # TODO: make depend on cooling factor
        self.temp *= 0.9
