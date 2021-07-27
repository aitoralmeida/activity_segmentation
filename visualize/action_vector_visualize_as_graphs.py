import json
import sys

from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

import numpy as np
import pandas as pd

# FILE with vectors
VECTORS = "data/2008-02-25_day.vector"
# FILE with labels
LABELS = "data/labels"
# FILE with edges
EDGES = "data/actions_locations_context.edgelist"

def main(argv):
    print(('*' * 20))
    print('Loading vectors...')
    sys.stdout.flush()
    # vectors
    vectors = pd.read_csv(VECTORS, header=None, sep=' ')
    # labels
    labels_file = open(LABELS, "r")
    labels = labels_file.read().splitlines()
    # edges
    edges_file = open(EDGES, "r")
    edges = edges_file.read().splitlines()
    # fit TSNE to reduce dim of embeddings
    print('Applying TSNE...')
    tsne = TSNE(n_components=2, perplexity=5, learning_rate=10, n_iter=1000, verbose=2)
    reduced_embeddings = tsne.fit_transform(vectors.values)
    # plot embeddings with action names (TSNE) as GRAPH
    print('Plotting embeddings as GRAPH...')
    G = nx.Graph()
    # add actions as nodes
    for label in labels:
        G.add_node(label)
    # add edges from graph (location, activities)
    for pair_of_edges in edges:
        edge_1, edge_2 = pair_of_edges.split(" ")
        G.add_edge(edge_1, edge_2)
    # position nodes based on TSNE dim reduction
    pos = {}
    counter = 0
    for reduced_embedding in reduced_embeddings:
        pos[labels[counter]] = (reduced_embedding[0], reduced_embedding[1])
        counter += 1
    # draw and save the graph
    nx.draw(G, pos, with_labels=True)
    plt.axis('equal')
    plt.show(block=False)
    plt.savefig('plots/graph/' + VECTORS[5:-7] + '_graph_' + EDGES[5:-8] + 'png', format="PNG")


if __name__ == "__main__":
    main(sys.argv)
