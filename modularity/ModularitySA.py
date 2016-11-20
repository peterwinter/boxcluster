from .base import modularity
from .base import BaseModularitySearch
import random
import copy
import math
from . import NestedSA as MNSA

limit = 25


class ModularitySAProblem(BaseModularitySearch):

    def __init__(self, G, Ti=1.0, Tf=0.005, Ts=0.995, f=1, fac=0.10):
        '''Sets the initial state for the problem.  The state is
        a list of modules, each containing a list of nodes.
        We should include the giant component (or current partition)
        of a graph such that there is one connected component
        '''
        self.G = self.graph_check(G)
        self.Ti = Ti
        self.Tf = Tf
        self.Ts = Ts
        if self.G is not None:
            self.global_net = copy.deepcopy(G)
            self.initialize_search(G, f, fac)
        num_nodes = len(G)
        self.n_individual_moves = self.calc_individual_move_n(fac, num_nodes)
        self.n_collective_moves = self.calc_collective_move_n(fac, num_nodes)

    def initialize_search(self, G, f, fac):
        # select a random number of partitions to begin with (range 2-7)
        p = 2  # random.choice(range(2,8))
        # initially we assume that all nodes are in the same module
        self.initial = self.get_random_partition(G, p)
        self.modules = copy.deepcopy(self.initial)
        self.L = G.size()
        subgraphs = self.get_subgraphs(self.modules)
        self.current_modularity = modularity(subgraphs, self.global_net, self.L)
        self.f = f
        self.fac = fac
        self.most_recent_moves = None

    def anneal(self):
        ''' While temperature T is less than final temperature
        (and convergence has not been reached)
        at each energy step take a successor and evaluate it's
        relative improvement using metropolis-style acceptance criteria.
        '''
        T = self.Ti
        count = 0
        while T > self.Tf and count <= limit:
            self.sort_modules()
            old_modules = copy.deepcopy(self.modules)
            move_modules = self.successor(T)
            # TODO: remove this
            if move_modules is None:
                print('move modules is none')
                continue
            if old_modules is None:
                print('old modules is none')
                continue
            try:
                new_energy = self.cost(move_modules)
                old_energy = self.cost(old_modules)
            except:
                print('why did this happen')
            # delta = new_energy - old_energy # was this
            delta = (new_energy - old_energy) / old_energy
            if new_energy > old_energy:
                self.modules = move_modules
                self.current_modularity = new_energy
            elif new_energy == old_energy:
                count += 1
                self.modules = old_modules
                self.current_modularity = old_energy
            else:
                p = math.exp(delta / T)
                if random.random() <= p:
                    # print "accept with p:" + str(math.exp(delta/T))
                    # print "p = " + str(p)
                    self.modules = move_modules
                    self.current_modularity = new_energy
                else:
                    self.modules = old_modules
                    self.current_modularity = old_energy
            # sanity check
            total_nodes = 0
            for s in self.modules:
                total_nodes += len(s)
            if total_nodes != len(self.G):
                print("Node mismatch!")
                print(self.modules)
                print(self.most_recent_moves)
                for i in range(len(self.modules)):
                    for j in range(i + 1, len(self.modules)):
                        if not set(self.modules[i]).isdisjoint(
                                set(self.modules[j])):
                            intersection = set(self.modules[i]) & set(
                                self.modules[j])
                            for k in intersection:
                                self.modules[i].remove(k)
                print(self.modules)
                #check for empty modules
            for s in self.modules:
                if len(s) == 0:
                    self.modules.remove(s)
            self.current_modularity = self.value()
            T *= self.Ts

    def value(self):
        '''Evaluate the cost of the current list of modules.'''
        return self.cost(self.modules)

    def cost(self, modules):
        '''Compute the modularity measure (as in Newman)
        given a set of partitions.'''

        sub = self.get_subgraphs(modules)
        try:
            return modularity(sub, self.global_net, self.L)
        except:
            print('BREAK')
            print(modules)
            print(sub)
            print(self.global_net)
            print(self.L)
            print(modularity)

    def successor(self, T):
        '''For each temperature T, we define fS**2 single node movements,
        and fS collective movements where a collected movement is either
        the merging of two modules or the splitting of a single module.'''

        # get the number of possible successors:
        # Single Node Part of Problem
        if len(self.modules) > 1:
            # Do the single node movements
            self.modules = self.single_moves(T=T)
        self.collective_moves(T=T)

    def collective_moves(self, T):
        # Do the collective movements
        n_collective_moves = self.n_collective_moves
        collective_movements = []
        H = copy.deepcopy(self.modules)
        for i in range(n_collective_moves):
            # select whether to merge or split
            collective_move_type = random.choice(["merge", "split"])
            if len(self.modules) == 1:
                collective_move_type = "split"
            if collective_move_type == "merge":
                merge_move = self.get_merge_move(H)
                if self.collective_accepted(merge_move, T):
                    # collective_movements.append(merge_move)
                    H = merge_move["result"]
            elif collective_move_type == "split":
                split_move = self.get_split_move(H)
                if self.collective_accepted(split_move, T):
                    # collective_movements.append(split_move)
                    H = split_move["result"]
            # check for empty modules
            for s in H:
                if len(s) == 0:
                    H.remove(s)
        return H

    def get_split_move(self, modules=None):
        '''Uses simulated annealing to split a randomly selected module.
        Annealing is provided by NestedSA.py'''
        if not modules:
            modules = self.modules
        # select a module to split
        module_to_split = random.choice(modules)
        # if there are no edges, propose a move instead
        if self.G.subgraph(module_to_split).size() == 0:
            return self.get_merge_move()
        module_index = modules.index(module_to_split)
        old_modules = copy.deepcopy(modules)
        sa = MNSA.ModularityNestedSA(self.G.subgraph(module_to_split), self.G)
        sa.anneal()
        new_module_graphs = sa.modules
        new_modules = []
        # print "after split:"
        for m in new_module_graphs:
            new_modules.append(m)
        result_modules = copy.deepcopy(modules)
        result_modules.pop(module_index)
        result_modules += new_modules
        # print result_modules
        modules = old_modules
        move = {"type": "split",
                "split": module_to_split,
                "new": new_modules,
                "result": result_modules}
        return move

    def get_merge_move(self, modules=None):
        '''Select to modules at random and merge them.'''
        if not modules:
            modules = self.modules
        if len(modules) == 1:
            return self.get_split_move(modules)
        # select two modules at random
        module_list_copy = copy.deepcopy(modules)
        module_a = random.choice(module_list_copy)
        module_b = random.choice(module_list_copy)
        while module_b == module_a:
            module_b = random.choice(module_list_copy)
        module_list_copy.remove(module_a)
        module_list_copy.remove(module_b)
        new_module = module_a + module_b
        # sanity check
        if len(module_a) + len(module_b) != len(new_module):
            print("Merge error, nodes lost!")
            print(module_a)
            print(module_b)
            print(new_module)
        module_list_copy.append(new_module)
        move = {"type": "merge",
                "grown": module_a,
                "removed": module_b,
                "result": module_list_copy}
        return move

    def collective_accepted(self, move, T):
        '''Tests to see if module-wide moves (merges and splits) are accepted under
        metropolis conditions.'''
        new_modularity = self.cost(move["result"])
        old_modularity = self.cost(self.modules)

        if new_modularity > old_modularity:
            return True
        else:
            if random.random() <= math.exp((new_modularity - old_modularity) / T):
                return True
            else:
                return False

        return False

    def clean_move_list(self, move_list):
        '''Given a list of single-node movements, removes any impossible moves from the
        set (if it were processed as an iterative list of instructions).'''
        if len(move_list) > 1:
            R = len(move_list)
            i = 0
            while i < R:
                j = 0
                while j < R:
                    print("i:" + str(i) + " j:" + str(j) + " R:" + str(R))
                    if i != j:
                        if move_list[i]["node"] == move_list[j][
                                "node"] and move_list[i]["from"] == move_list[
                                    j]["from"]:
                            move_list.pop(j)
                        elif move_list[i]["node"] == move_list[j][
                                "node"] and move_list[i]["to"] == move_list[j][
                                    "to"]:
                            move_list.pop(j)
                    j += 1
                    R = len(move_list)
                i += 1
                if i == R:
                    return move_list
        return move_list


    def get_random_partition(self, G, num_modules=2):
        '''Selects a partitioning (into num_modules partitions) at random of
        the NetworkX graph G.'''
        # select two random subgraphs of G
        modules = dict()
        for i in range(num_modules - 1):
            modules.setdefault(i, [])
        all_nodes = G.nodes()
        nodes_to_remove = int((num_modules - 1) * len(G) / num_modules)
        for i in range(nodes_to_remove):
            node = random.choice(all_nodes)
            s = random.choice(range(num_modules - 1))
            modules[s].append(node)
            all_nodes.remove(node)
        partitions = []
        for s in modules.keys():
            partitions.append(modules[s])
        partitions.append(all_nodes)
        return partitions
