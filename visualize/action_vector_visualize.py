import json
import sys

from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# FILE with vectors
VECTORS = "data/2008-02-25_day.vector"
# FILE with labels
LABELS = "data/labels"

def plot_embeddings_reduced(embeddings, actions, filename):
    fig, ax = plt.subplots()
    ax.scatter(embeddings[:, 0], embeddings[:, 1])
    for i, txt in enumerate(actions):
        ax.annotate(actions[i], (embeddings[:, 0][i], embeddings[:, 1][i]))
    ax.axis('equal')
    fig.savefig(filename)

def main(argv):
    print(('*' * 20))
    print('Loading vectors...')
    sys.stdout.flush()
    # vectors
    vectors = pd.read_csv(VECTORS, header=None, sep=' ')
    # labels
    labels_file = open(LABELS, "r")
    labels = labels_file.readlines()
    # fit TSNE to reduce dim of embeddings
    print('Applying TSNE...')
    tsne = TSNE(n_components=2, perplexity=5, learning_rate=10, n_iter=500, verbose=2)
    reduced_embeddings = tsne.fit_transform(vectors.values)
    # plot embeddings with action names (TSNE)
    print('Plotting embeddings...')
    plot_embeddings_reduced(reduced_embeddings, labels, 'plots/' + VECTORS[5:-6] + 'png')

if __name__ == "__main__":
    main(sys.argv)
