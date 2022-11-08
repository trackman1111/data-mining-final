import networkx as nx
import numpy as np

import node2vec
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def read_graph():
    G = nx.read_edgelist('nfl.edgelist', nodetype=str, create_using=nx.DiGraph())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    G = G.to_undirected()

    return G


def learn_embeddings(walks):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, vector_size=128, window=5, min_count=1, workers=8)
    model.save('nfl.model')
    return model


def main():
    nx_G = read_graph()
    G = node2vec.Graph(nx_G, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=500, walk_length=100)
    model = learn_embeddings(walks)
    print(model.wv.most_similar("WAS"))
    nodes = [x for x in list(model.wv.key_to_index.keys())]
    embeddings = np.array([model.wv[x] for x in nodes])

    tsne = TSNE(n_components=2, random_state=7, perplexity=15)
    embeddings_2d = tsne.fit_transform(embeddings)

    figure = plt.figure(figsize=(11, 9))
    ax = figure.add_subplot(111)

    for i, word in enumerate(nodes):
        plt.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]))

    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

    plt.savefig('test.png')


if __name__ == "__main__":
    main()
