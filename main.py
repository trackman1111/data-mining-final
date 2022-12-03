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
    walks = G.simulate_walks(num_walks=20000, walk_length=100)
    model = learn_embeddings(walks)
    # print(model.wv.most_similar("WAS"))
    # print(model.wv.most_similar("TEN"))

    nodes = [x for x in list(model.wv.key_to_index.keys())]
    embeddings = np.array([model.wv[x] for x in nodes])

    tsne = TSNE(n_components=2, random_state=7, perplexity=15)
    embeddings_2d = tsne.fit_transform(embeddings)

    figure = plt.figure(figsize=(11, 9))
    ax = figure.add_subplot(111)

    for i, word in enumerate(nodes):
        plt.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]))

    colors = []
    nfc_east = ['WAS', 'NYG', 'PHI', 'DAL']
    nfc_north = ['GB', 'MIN', 'DET', 'CHI']
    nfc_south = ['NO', 'TB', 'CAR', 'ATL']
    nfc_west = ['LA', 'SF', 'ARI', 'SEA']
    afc_east = ['MIA', 'NYJ', 'NE', 'BUF']
    afc_north = ['CIN', 'PIT', 'CLE', 'BAL']
    afc_south = ['HOU', 'TEN', 'IND', 'JAX']

    for node in nodes:
        if node in nfc_east:
            colors.append('green')
        elif node in nfc_north:
            colors.append('blue')
        elif node in nfc_south:
            colors.append('red')
        elif node in nfc_west:
            colors.append('teal')
        elif node in afc_east:
            colors.append('orange')
        elif node in afc_north:
            colors.append('purple')
        elif node in afc_south:
            colors.append('yellow')
        else:
            colors.append('pink')

    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors)

    plt.savefig('scatter.png')


if __name__ == "__main__":
    main()
