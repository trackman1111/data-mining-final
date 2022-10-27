class Node2Vec:

    # p -- return rate
    # q -- exploration rate
    # random_walk_lengths -- number of steps taken in each random walk
    # num_walks -- number of random walks completed.

    def __init__(self, node_graph, q, p, random_walk_length, num_walks):
        self.node_graph = node_graph
        self.q = q
        self.p = p
        self.random_walk_length = random_walk_length
        self.num_walks = num_walks

    def random_walk(self):
        pass
        # at each node, new exploration should be 1/q, return to previous node should be 1/p
