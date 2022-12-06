import numpy as np
import random


class Node2Vec:

    def __init__(self, graph, p, q):
        self.alias_edges = None
        self.alias_nodes = None
        self.graph = graph
        self.p = p
        self.q = q

    def random_walk(self, walk_length, start_node):
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            neighbors = list(self.graph.neighbors(cur))
            if len(neighbors) > 0:
                if len(walk) == 1:
                    walk.append(neighbors[alias_draw(self.alias_nodes[cur][0], self.alias_nodes[cur][1])])
                else:
                    prev_step = walk[-2]
                    next_step = neighbors[alias_draw(self.alias_edges[(prev_step, cur)][0],
                                                     self.alias_edges[(prev_step, cur)][1])]
                    walk.append(next_step)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        walks = []
        nodes = list(self.graph.nodes())
        for _ in range(num_walks):
            for node in nodes:
                walks.append(self.random_walk(walk_length, node))
        return walks

    def calculate_edges(self, src, dst):
        probs = []
        for neighbors in self.graph.neighbors(dst):
            if neighbors == src:
                probs.append(self.graph[dst][neighbors]['weight'] / self.p)
            elif self.graph.has_edge(neighbors, src):
                probs.append(self.graph[dst][neighbors]['weight'])
            else:
                probs.append(self.graph[dst][neighbors]['weight'] / self.q)
        total_prob = sum(probs)
        normalized_probs = [float(prob) / total_prob for prob in probs]

        return alias_setup(normalized_probs)

    def preprocess_edges(self):

        alias_edges = {}

        for edge in self.graph.edges():
            alias_edges[edge] = self.calculate_edges(edge[0], edge[1])
            alias_edges[(edge[1], edge[0])] = self.calculate_edges(edge[1], edge[0])

        self.alias_edges = alias_edges

    def preprocess_nodes(self):
        alias_nodes = {}
        for node in self.graph.nodes():
            probs = [self.graph[node][neighbor]['weight'] for neighbor in self.graph.neighbors(node)]
            total_prob = sum(probs)
            normalized_probs = [float(prob) / total_prob for prob in probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        self.alias_nodes = alias_nodes


def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1
        if q[large] < 1:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(j, q):
    i = int(random.uniform(0, len(j)))
    if np.random.rand() < q[i]:
        return i
    else:
        return j[i]
