import sys
import argparse

import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

def main(argv):
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/activity_segmentation/kasteren_house_a/reduced",
                        nargs="?",
                        help="Dataset dir")
    parser.add_argument("--dataset_file",
                        type=str,
                        default="base_kasteren_reduced.csv",
                        nargs="?",
                        help="Dataset file")
    parser.add_argument("--word2vec_window",
                        type=int,
                        default=5,
                        nargs="?",
                        help="Word2vec window")
    parser.add_argument("--iterations",
                        type=int,
                        default=5,
                        nargs="?",
                        help="Word2vec iterations")
    parser.add_argument("--embedding_size",
                        type=int,
                        default=50,
                        nargs="?",
                        help="Word2vec embedding size")
    args = parser.parse_args()
    # loading full dataset
    print('Loading DATASET...')
    DATASET = args.dataset_dir + "/" + args.dataset_file
    df_har = pd.read_csv(DATASET, parse_dates=[[0, 1]], index_col=0, sep=' ', header=None)
    df_har.columns = ['sensor', 'action', 'event', 'activity']
    df_har.index.names = ["timestamp"]
    print('DATASET loaded')
    # retrieve actions
    actions = df_har['action']
    # create word2vec model
    model = Word2Vec([actions.values.tolist()],
            size=args.embedding_size, window=args.word2vec_window, min_count=0, 
            iter=args.iterations,
            workers=multiprocessing.cpu_count())
    # write model to vector file
    vector_file_name = args.dataset_dir + '/word2vec_models/word2vec_embedding_size_' + str(args.embedding_size) + '_iterations_' + str(args.iterations) + '_word2vec_window_' + str(args.word2vec_window)
    model.wv.save_word2vec_format(vector_file_name, binary=False)

if __name__ == "__main__":
    main(sys.argv)