import networkx as nx
import numpy as np

import node2vec
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def create_weighted_graph():
    graph = nx.read_edgelist('nfl.edgelist', nodetype=str, create_using=nx.DiGraph())
    for edge in graph.edges():
        graph[edge[0]][edge[1]]['weight'] = 1
    graph = graph.to_undirected()

    return graph


def learn_embeddings(walks):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, vector_size=32, window=5, min_count=1, workers=8)
    model.save('nfl.model')
    return model


def main():
    nx_graph = create_weighted_graph()
    graph = node2vec.Node2Vec(nx_graph, p=1, q=1)
    graph.preprocess_edges()
    graph.preprocess_nodes()
    walks = graph.simulate_walks(num_walks=300, walk_length=10)
    model = learn_embeddings(walks)

    nodes = [x for x in list(model.wv.key_to_index.keys())]
    embeddings = np.array([model.wv[x] for x in nodes])

    tsne = TSNE(n_components=2, random_state=7, perplexity=15)
    embeddings_2d = tsne.fit_transform(embeddings)

    figure = plt.figure(figsize=(11, 9))
    ax = figure.add_subplot(111)

    for i, word in enumerate(nodes):
        plt.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]))

    colors = []
    divisions = {'nfc_east': ['WAS', 'NYG', 'PHI', 'DAL'], 'nfc_north': ['GB', 'MIN', 'DET', 'CHI'],
                 'nfc_south': ['NO', 'TB', 'CAR', 'ATL'], 'nfc_west': ['LA', 'SF', 'ARI', 'SEA'],
                 'afc_east': ['MIA', 'NYJ', 'NE', 'BUF'], 'afc_north': ['CIN', 'PIT', 'CLE', 'BAL'],
                 'afc_south': ['HOU', 'TEN', 'IND', 'JAX'], 'afc_west': ['LAC', 'KC', 'DEN', 'LV']}

    count = 0
    correct = 0
    for key in divisions:
        for team in divisions[key]:
            most_similar = model.wv.most_similar(team)
            for x in range(0, 3):
                if most_similar[x][0] in divisions[key]:
                    correct += 1
                count += 1
    print("Correct Top 3 Relations: " + str(correct) + "/" + str(count))

    for node in nodes:
        if node in divisions['nfc_east']:
            colors.append('green')
        elif node in divisions['nfc_north']:
            colors.append('blue')
        elif node in divisions['nfc_south']:
            colors.append('red')
        elif node in divisions['nfc_west']:
            colors.append('teal')
        elif node in divisions['afc_east']:
            colors.append('orange')
        elif node in divisions['afc_north']:
            colors.append('purple')
        elif node in divisions['afc_west']:
            colors.append('yellow')
        else:
            colors.append('pink')

    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors)

    plt.savefig('scatter.png')


if __name__ == "__main__":
    main()
