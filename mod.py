import networkx as nx
import random
import numpy as np
import copy as cp
from collections import abc
from collections import OrderedDict
from .mixins import FitnessEqualitiesMixin

from collections import namedtuple

SingleMove = namedtuple('SingleMove', ('node', 'from_mod', 'to_mod'))


class BaseModules(abc.Iterable, FitnessEqualitiesMixin):
    """ basic class functionality """

    def __init__(self):
        self.fitness = -np.inf
        self.nodes = {}
        self.mods = OrderedDict()

    def __getitem__(self, node):
        return self.nodes[node]

    def __setitem__(self, node, mod):
        try:
            self.mods[mod] |= set([node])
        except KeyError:
            self.mods[mod] = set([node])
        self.nodes[node] = mod

    def __delitem__(self, node):
        mod = self.nodes[node]
        del self.nodes[node]
        self.mods[mod] -= set([node])

    def __len__(self):
        return len(self.mods)

    def __iter__(self):
        for mod in self.mods.values():
            yield list(mod)

    def items(self):
        for mod, nodes in self.mods.items():
            yield mod, nodes

    def __repr__(self):
        f = round(self.fitness, ndigits=2)
        m = [vals for vals in self.mods.values()]
        return 'Fit:{f} modules: {b}'.format(f=f, b=m)

    def insert(self, node, mod):
        self.mods[mod] = node
        self.nodes[node] = mod


class SimpleModules(BaseModules):
    """ Add in next level of functionality"""

    def __init__(self, modules=[], mods=None, nodes=None, fitness=-np.inf):
        """either pass in modules or mods and nodes """
        self.fitness = fitness
        if modules:
            self.nodes = {}
            self.mods = OrderedDict()
            for (i, m) in enumerate(modules):
                if not m:
                    continue
                self.mods[i] = set(m)
                for node in m:
                    self.nodes[node] = i
                    # add to set
        elif mods is not None and nodes is not None:
            self.mods = mods
            self.nodes = nodes

    def move_node(self, node, mod):
        old_mod = self.nodes[node]
        self.mods[old_mod] -= set([node])
        if not self.mods[old_mod]:
            del self.mods[old_mod]
            self.reindex_mods()
        self.__setitem__(node=node, mod=mod)

    def reindex_mods(self):
        # print('reindex')
        new_mods = OrderedDict()
        for i, (mod_key, mod) in enumerate(self.mods.items()):
            new_mods[i] = mod
            # need to reindex nodes too
            if i != mod_key:
                for m in mod:
                    self.nodes[m] = i
        self.mods = new_mods

    def copy(self):
        mods = cp.deepcopy(self.mods)
        nodes = cp.deepcopy(self.nodes)
        return self.__class__(mods=mods, nodes=nodes, fitness=self.fitness)

    def merge(self, mod1, mod2):
        """ merges all nodes from mod2 into mod1 """
        mod2_nodes = self.mods.pop(mod2)
        self.mods[mod1] |= mod2_nodes
        for node in mod2_nodes:
            self.nodes[node] = mod1
        self.reindex_mods()

    def split_options(self):
        split_choices = []
        for i, mod_nodes in self.items():
            if len(mod_nodes) > 1:
                split_choices.append(i)
        return split_choices

    def split(self, mod, split_nodes):
        assert isinstance(split_nodes, set)
        mod_nodes = self.mods[mod]
        # check that split is not full module
        err = 'SplitError: split too large'
        assert len(split_nodes) < len(mod_nodes), err
        # check that all split nodes are contained in chosen_module
        missing_nodes = split_nodes - mod_nodes
        err = 'SplitError: {miss} nodes not in module {mod}'
        assert len(missing_nodes) == 0, err.format(miss=missing_nodes, mod=mod)
        # change self.mods to reflect split
        self.mods[mod] = mod_nodes - split_nodes
        new_mod = max(self.mod_list()) + 1
        self.mods[new_mod] = split_nodes
        # change self.nodes to reflect split
        for node in split_nodes:
            self[node] = new_mod

    def mod_list(self):
        return list(self.mods.keys())


class Modules(SimpleModules):
    """Add in high level module functions"""

    # TODO: subgraphs is wonky
    def subgraphs(self, G):
        '''produces list of subgraphs where each subgraphs only contains
        nodes from one module.'''
        subgraphs = []
        for mod in self:
            # print(mod)
            if mod:
                m = list(mod)
                sub = G.subgraph(m)
                subgraphs.append(sub)
        return subgraphs

    def calculate_fitness(self, G, L):
        M = 0
        for mod in self:
            l_s = len(G.subgraph(mod).edges())
            d_s = sum(d for _, d in G.degree_iter(nbunch=mod))
            M += (l_s / L) - (d_s / (2.0 * L))**2
        self.fitness = M
        return M

    def move_random_single(self):
        node_to_move = random.choice(list(self.nodes.keys()))
        current_mod = self[node_to_move]
        # print('node', node_to_move)
        # print('current', current_mod)
        list_of_module_indices = self.mod_list()
        # print('mod inds', list_of_module_indices)
        # print('mods', self.mods)
        # print('nodes', self.nodes)
        list_of_module_indices.remove(current_mod)
        new_mod = random.choice(list_of_module_indices)
        self.move_node(node=node_to_move, mod=new_mod)

    def move_random_merge(self):
        mods = self.mod_list()
        if len(mods) >= 2:
            mod1, mod2 = np.random.choice(mods, size=2, replace=False)
            self.merge(mod1, mod2)
            return True
        else:
            return False

    def move_random_split(self):
        split_options = self.split_options()
        # if no splits possible
        if not split_options:
            return False
        mod = np.random.choice(split_options)
        mod_nodes = list(self.mods[mod])
        mod_size = len(mod_nodes)
        split_size = np.random.randint(low=1, high=mod_size)
        split_nodes = np.random.choice(mod_nodes, size=split_size)
        split_nodes = set(split_nodes)
        self.split(mod=mod, split_nodes=split_nodes)
        return True

    @classmethod
    def random_initialization(cls, G, L=None, num_modules=3):
        all_nodes = G.nodes()
        if L is None:
            L = len(all_nodes)
        partition = np.random.randint(num_modules, size=L, dtype=int)
        nodes = {n: p for (n, p) in zip(all_nodes, partition)}
        mods = OrderedDict()
        for i in range(num_modules):
            mod_nodes = np.array(all_nodes)[partition == i]
            mods[i] = set(mod_nodes)
        modules = cls(mods=mods, nodes=nodes)
        modules.calculate_fitness(G, L=L)
        return modules
