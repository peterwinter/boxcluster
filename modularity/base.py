import networkx as nx
import random
import copy
import math
# from collections import namedtuple


def largest_connected(G):
    return next(nx.connected_component_subgraphs(G))


def modularity(modules, G, L):
    """ calculate modularity
    modularity = [list of nx.Graph objects]
    G = graph
    L = num of links
    """
    N_m = len(modules)
    M = 0.0
    for s in range(N_m):
        l_s = 0.0
        d_s = 0
        for i in modules[s]:
            l_s += float(modules[s].degree(i))
            d_s += float(G.degree(i))
        M += (l_s / L) - (d_s / (2.0 * L))**2
    return M


# TODO
class Modules(object):

    def __init__(self, ):
        self.items = []
        self.modularity = 0

    def sort(self):
        ''' Sorts the node indices for each modules in self.modules in
        descending order.
        '''
        for s in self.modules:
            s.sort()


class BaseModularityClass(object):
    # self.single_Move = namedtuple('Move', ('node', 'from', 'to'))

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


    # TODO: make moves happen as they are chosen...
    def clean_move_tuples(self, move_list):
        '''Remove impossible moves from a list of triples.
        Assumes moves occur iteratively.'''
        move_tuples = self.moves_to_tuples(move_list)
        allowable_moves = []
        for m in move_tuples:
            if m not in allowable_moves:
                allowable_moves.append(m)
        return self.tuples_to_moves(allowable_moves)

    def moves_to_tuples(self, move_list):
        '''Convert a list of move dictionaries to triples.'''
        move_tuples = []
        for m in move_list:
            move_tuples.append((m["node"], m["from"], m["to"]))
        return move_tuples

    def tuples_to_moves(self, t_list):
        '''Convert a list of triples to move dictionaries.'''
        moves = []
        for t in t_list:
            moves.append({"node": t[0], "from": t[1], "to": t[2]})
        return moves

class BaseModularitySearch(BaseModularityClass):

    def calc_individual_move_n(self, fac, num_nodes):
        if (fac * num_nodes**2 < 10):
            num_individual_moves = 10
        else:
            num_individual_moves = int(math.floor(fac * num_nodes**2))
        return num_individual_moves

    def calc_collective_move_n(self, fac, num_nodes):
        if (self.fac * num_nodes < 2):
            n_collective_moves = 2
        else:
            n_collective_moves = int(math.floor(self.fac * num_nodes))
        return n_collective_moves

    def accepted(self, move, T):
        #check the modularity
        H = []
        module_nodes_from = copy.deepcopy(self.modules[move["from"]])
        module_nodes_to = copy.deepcopy(self.modules[move["to"]])
        module_nodes_from.remove(move["node"])
        module_nodes_to.append(move["node"])
        for i in range(len(self.modules)):
            if i != move["from"] and i != move["to"]:
                H.append(self.modules[i])
            elif i == move["from"]:
                H.append(module_nodes_from)
            elif i == move["to"]:
                H.append(module_nodes_to)
        new_modularity = self.cost(self.get_subgraphs(H))
        if new_modularity > self.current_modularity or T == 0:
            return True
        else:
            p = math.exp((new_modularity - self.current_modularity) / T)
            if random.random() <= p:
                return True
            else:
                return False
        return False


    def get_single_node_move(self):
        '''Select two nodes at random to swap.'''
        # select a node at random
        node_to_move = random.choice(self.G.nodes())
        # identify the partition the node is in
        list_of_module_indices = list(range(len(self.modules)))
        for i in list_of_module_indices:
            if node_to_move in self.modules[i]:
                currently_in = i
        list_of_module_indices.remove(currently_in)
        # choose a random partition to move to
        new_partition = random.choice(list_of_module_indices)
        move = {"node": node_to_move,
                "from": currently_in,
                "to": new_partition}
        return move


    def single_moves(self, T):
        single_node_movements = []
        for i in range(self.n_individual_moves):
            move = self.get_single_node_move()
            if self.accepted(move, T):
                single_node_movements.append(move)
        single_node_movements = self.clean_move_tuples(single_node_movements)
        self.most_recent_moves = single_node_movements
        return self.do_single_node_movements(single_node_movements)


    def do_single_node_movements(self, move_set):
        '''Given a list of single node movements as a list of dictionaries
        structured: "node","from","to", compute the resulting set of modules.
        '''
        # for each module, find the set of nodes added
        # and the set of nodes removed from it
        temporary_modules = copy.deepcopy(self.modules)
        for m in move_set:
            condition1 = m["node"] not in temporary_modules[m["to"]]
            condition2 = m["node"] in temporary_modules[m["from"]]
            # TODO: check if it should be this
            # if condition1 and condition2:
            if condition1:
                temporary_modules[m["to"]].append(m["node"])
            if condition2:
                temporary_modules[m["from"]].remove(m["node"])
        return temporary_modules


    def sort_modules(self):
        ''' Sorts the node indices for each modules in self.modules in
        descending order.
        '''
        for s in self.modules:
            s.sort()
