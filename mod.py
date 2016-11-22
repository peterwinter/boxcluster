import networkx as nx
import random
import numpy as np
import copy as cp
from collections import abc
from collections import OrderedDict

from collections import namedtuple

SingleMove = namedtuple('SingleMove', ('node', 'from_mod', 'to_mod'))


def largest_connected(G):
    return next(nx.connected_component_subgraphs(G))


class BaseModules(abc.Iterable):
    """ basic class functionality """

    def __init__(self, modules=[], mods=None, nodes=None):
        """either pass in modules or mods and nodes """
        self.fitness = -np.inf

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
        new_mods = OrderedDict()
        for i, (mod_key, mod) in enumerate(self.mods.items()):
            new_mods[i] = mod
        self.mods = new_mods

    def copy(self):
        mods = cp.deepcopy(self.mods)
        nodes = cp.deepcopy(self.mods)
        return self.__class__(mods=mods, nodes=nodes)

    def __getitem__(self, node):
        return self.nodes[node]

    def __setitem__(self, node, mod):
        self.mods[mod] |= set([node])
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

    def __repr__(self):
        f = round(self.fitness, ndigits=2)
        m = [vals for vals in self.mods.values()]
        return 'Fit:{f} modules: {b}'.format(f=f, b=m)

    def insert(self, node, mod):
        self.mods[mod] = node
        self.nodes[node] = mod

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __le__(self, other):
        return self.fitness <= other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __ne__(self, other):
        return self.fitness != other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __ge__(self, other):
        return self.fitness >= other.fitness

    def merge(self, mod1, mod2):
        """ merges all nodes from mod2 into mod1 """
        mod2_nodes = self.mods.pop(mod2)
        self.mods[mod1] |= mod2_nodes
        for node in mod2_nodes:
            self.nodes[node] = mod1
        self.reindex_mods()

    def split(self, mod, split_nodes):
        assert isinstance(split_nodes, set)


    def mod_list(self):
        return list(self.mods.keys())

class Modules(BaseModules):

    # TODO: subgraphs is wonky
    def subgraphs(self, G):
        '''produces list of subgraphs where each subgraphs only contains
        nodes from one module.'''
        subgraphs = []
        for mod in self:
            print(mod)
            if mod:
                m = list(mod)
                sub = G.subgraph(m)
                subgraphs.append(sub)
        return subgraphs

    def set_modularity(self, G, L):
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
        list_of_module_indices = self.mod_list()
        list_of_module_indices.remove(current_mod)
        new_mod = random.choice(list_of_module_indices)
        self.move_node(node=node_to_move, mod=new_mod)

    def move_random_merge(self):
        mods = self.mod_list()
        if len(mods) >= 2:
            mod1, mod2 = np.random.choice(mods, size=2, replace=False)
        self.merge(mod1, mod2)
