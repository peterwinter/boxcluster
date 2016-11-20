import random
import copy
import math
from .base import modularity
from .base import BaseModularitySearch

limit = 25


# specifically for finding split moves
class ModularityNestedSA(BaseModularitySearch):

    # unlike ModularitySA, this also contains global_net
    def __init__(self, G, global_net, Ti=1.0, Tf=0.005, Ts=0.995, f=1, fac=0.10):
        G = self.graph_check(G)
        if G is None:
            return None
        self.G = G
        self.global_net = global_net
        self.Ti = Ti
        self.Tf = Tf
        self.Ts = Ts

        self.initial = self.get_random_partition(G)
        self.modules = copy.deepcopy(self.initial)
        self.L = G.size()

        self.S = len(G)
        self.current_modularity = modularity(
            self.get_subgraphs(self.modules), self.global_net, self.L)
        self.f = f
        self.fac = fac

        num_nodes = len(G)
        self.n_individual_moves = self.calc_individual_move_n(fac, num_nodes)

    def anneal(self):
        T = self.Ti
        count = 0
        while T > self.Tf and count <= limit:
            old_modules = copy.deepcopy(self.modules)
            move_modules = self.successor(T)
            new_energy = self.cost(self.get_subgraphs(move_modules))
            old_energy = self.cost(self.get_subgraphs(old_modules))
            delta = new_energy - old_energy
            if new_energy > old_energy:
                self.modules = move_modules
                self.current_modularity = new_energy
            elif new_energy == old_energy:
                count += 1
                self.modules = old_modules
                self.current_modularity = old_energy
            else:
                if random.random() <= math.exp(delta / T):
                    #                    print "accept with p:" + str(math.exp(delta/T))
                    self.modules = move_modules
                    self.current_modularity = new_energy
                else:
                    self.modules = old_modules
                    self.current_modularity = old_energy
#            print "T:" + str(T)
#            print "new_energy:" + str(new_energy)
#            print "old_energy:" + str(old_energy)
#            print "change in energy:" + str(new_energy - old_energy)
            T *= self.Ts

    def value(self):
        return self.cost(self.get_subgraphs(self.modules))

    def cost(self, modules):
        return modularity(modules, self.global_net, self.L)

    def successor(self, T):
        '''For each temperature T, we define fS**2 single node movements,
        and fS collective movements where a collected movement is either
        the merging of two modules or the splitting of a single module.
        '''
        # get the number of possible successors:
        return self.single_moves(T=T)

    def get_random_partition(self, G):
        # select two random subgraphs of G
        module_a = []
        all_nodes = G.nodes()
        for i in range(int(len(G) / 2)):
            node = random.choice(all_nodes)
            module_a.append(node)
            all_nodes.remove(node)

        modules = []
        modules.append(module_a)
        modules.append(all_nodes)

        return modules
