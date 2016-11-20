#! /usr/bin/env python
"""
Implementation of the GirvanNewman algorithm
for finding the best modularity in a network
"""
import copy
import random
import networkx as nx
from .base import modularity
from .base import largest_connected


def get_graph(filename):
    G = nx.Graph()
    f = open(filename)
    data = f.readlines()
    edges = []
    for line in data:
        entry = map(int, line.rstrip().split())
        if entry:
            edges.append(tuple(entry))
    G.add_edges_from(edges)
    f.close()
    return G


def make_inverse_dict(d):
    inverse_map = {}
    for k, v in d.items():
        inverse_map.setdefault(v, [])
        inverse_map[v].append(k)
    return inverse_map


def girvan_newman(G):
    ranked_betweenness_edges = nx.edge_betweenness(G)
    inverse_map = make_inverse_dict(ranked_betweenness_edges)

    max_edge_key = max(ranked_betweenness_edges.values())
    max_edge_choice = random.choice(range(len(inverse_map[max_edge_key])))
    max_edge = inverse_map[max_edge_key][max_edge_choice]
    G.remove_edge(max_edge[0], max_edge[1])

    if len(G.edges()) <= 2:
        return [max_edge] + G.edges()
    else:
        return [max_edge] + girvan_newman(G)


# def get_protected_edges(G, similar_sets):
#     protected_edges = []
#     for s in similar_sets:
#         for u in s:
#             for v in s[1:]:
#                 if G.has_edge(u, v):
#                     protected_edges.append((u, v))
#     return protected_edges


def find_unprotected_edge(ranks, d, max_edge, protected_edges):
    protected_edge = True
    while protected_edge:
        index = max(ranks)
        protected_edge = False
        while max_edge in protected_edges:
            protected_edge = True
            if len(d[index]) > 1:
                d[index].remove(max_edge)
            else:
                ranks.remove(max(ranks))
                index = max(ranks)
            max_edge = random.choice(d[index])
    if max_edge in protected_edges:
        print( "wrong!")
    return max_edge


def girvan_newman_partition(G, partition_count, protected_edges=None):
    ranked_betweenness_edges = nx.edge_betweenness(G)
    inverse_map = make_inverse_dict(ranked_betweenness_edges)

    max_edge_key = max(ranked_betweenness_edges.values())
    max_edge_choice = random.choice(range(len(inverse_map[max_edge_key])))
    max_edge = inverse_map[max_edge_key][max_edge_choice]

    if protected_edges:
        ranks = ranked_betweenness_edges.values()
        ranks.sort()
        ranks.reverse()
        if len(ranked_betweenness_edges.keys()) > len(protected_edges):
            #imperfect, but faster than computing set differences
            max_edge = find_unprotected_edge(ranks, copy.copy(inverse_map),
                                             max_edge, protected_edges)
        else:
            candidates = list(
                set(ranked_betweenness_edges.keys()).difference(
                    set(protected_edges)))
            if len(candidates) > 0:
                while max_edge not in candidates:
                    max_edge = find_unprotected_edge(
                        ranks, copy.copy(inverse_map), max_edge,
                        protected_edges)

    G.remove_edge(max_edge[0], max_edge[1])

    if nx.number_connected_components(G) >= partition_count:
        return [max_edge]
    else:
        return [max_edge] + girvan_newman_partition(G, partition_count,
                                                    protected_edges)



def get_strong_similarity_regions(similarity_matrix, threshold=1.0):
    # for the i-th row and j-th column, consider the
    # upper triangular matrix for "blocks" of similarlity
    similar_regions = []
    i = 0
    while i < len(similarity_matrix):
        region = [i]
        contiguous = 0
        for j in range(i + 1, len(similarity_matrix[0])):
            if similarity_matrix[i, j] >= threshold:
                region.append(j)
                contiguous += 1
        if len(region) > 0:
            similar_regions.append(region)
        i += 1 + contiguous
    return similar_regions


def get_similarity_matrix(similarity_matrix_file):
    sm_file = open(similarity_matrix_file)
    sm_raw_data = sm_file.readlines()
    sm_file.close()
    similarity_matrix = None
    #similarity_matrix = np.array(map(string.split,sm_raw_data), float)
    return similarity_matrix


def greedy_partitioning(G, P0=1, similarity_matrix=None):
    # get the total number of edges
    L = G.size()

    last_partitioning = nx.to_numpy_matrix(G)

    if P0 == 1:
        # get the giant component
        G = largest_connected(G)[0]
        best_modularity = modularity([G], L)
    else:
        best_modularity = modularity(largest_connected(G))

    current_modularity = copy.copy(best_modularity)
    number_of_partitions = P0

    while best_modularity <= current_modularity:
        if current_modularity >= best_modularity:
            best_modularity = copy.copy(current_modularity)
            number_of_partitions += 1
            removal_order = girvan_newman_partition(G, number_of_partitions)
            current_modularity = modularity(
                largest_connected(G), L)
            last_partitioning = nx.to_numpy_matrix(G)
            if len(removal_order) == 1:
                G.add_edge(removal_order[0][0], removal_order[0][1])
            else:
                G.add_edges_from(removal_order)

    partitions = nx.from_numpy_matrix(last_partitioning)

    return partitions


def main():
    #initialization: read parameters

    similarity_matrix_present = False

    #    optlist, args = getopt.getopt(sys.argv[1:], 'p:')
    #assume that the graph file is a set of edges as in Marta's subnet.dat files
    graph_file = "testnet.dat"  #args[0]

    #is a coclassification file present?
    #    for o,a in optlist:
    #        if o == "-p":
    #            similarity_matrix_present=True
    #            similarity_matrix_file = a

    G = get_graph(graph_file)
    # get the total number of edges
    L = G.size()
    # get the giant component
    G = list(largest_connected(G))[0]

    # similarity_matrix_file = "sample_similarity.dat"
    # similarity_matrix = get_similarity_matrix(similarity_matrix_file)
    # similarity_regions = get_strong_similarity_regions(similarity_matrix)
    # protected_edges = get_protected_edges(G, similarity_regions)
    # print( 'protected edges:')
    # print( protected_edges)
    number_of_partitions = 1
    best_modularity = modularity([G], L)
    current_modularity = copy.copy(best_modularity)
    last_partitioning = nx.to_numpy_matrix(G)

    while best_modularity <= current_modularity:
        if current_modularity >= best_modularity:
            best_modularity = copy.copy(current_modularity)
            number_of_partitions += 1
            removal_order = girvan_newman_partition(G, number_of_partitions)
            current_modularity = modularity(largest_connected(G), L)
            last_partitioning = nx.to_numpy_matrix(G)
            if len(removal_order) == 1:
                G.add_edge(removal_order[0][0], removal_order[0][1])
            else:
                G.add_edges_from(removal_order)

    partitions = nx.from_numpy_matrix(last_partitioning)
    print( "best modularity: " + str(best_modularity))
    print( "number of partitions:" + str(number_of_partitions))
    print( "partitions:")
    for p in largest_connected(partitions):
        print( "nodes:")
        print( p.nodes())
        print( "edges:")
        print( p.edges())


if __name__ == "__main__":
    main()
