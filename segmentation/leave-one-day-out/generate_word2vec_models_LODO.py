import sys
import argparse

import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

from sklearn.model_selection import LeaveOneOut

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
    parser.add_argument("--results_dir",
                        type=str,
                        default="generated_models",
                        nargs="?",
                        help="Results dir")
    parser.add_argument("--embedding_size",
                        type=int,
                        default=50,
                        nargs="?",
                        help="Embedding size for word2vec algorithm")
    parser.add_argument("--window_size",
                        type=int,
                        default=1,
                        nargs="?",
                        help="Window size for word2vec algorithm")
    parser.add_argument("--iterations",
                        type=int,
                        default=5,
                        nargs="?",
                        help="Iterations for word2vec algorithm")
    parser.add_argument("--min_sensor_events",
                        type=int,
                        default=10,
                        nargs="?",
                        help="Minimum number of sensor events for a day to be evaluated")
    args = parser.parse_args()
    # loading full dataset
    print('Loading DATASET...')
    DATASET = args.dataset_dir + "/" + args.dataset_file
    df_har = pd.read_csv(DATASET, parse_dates=[[0, 1]], index_col=0, sep=' ', header=None)
    df_har.columns = ['sensor', 'action', 'event', 'activity']
    df_har.index.names = ["timestamp"]
    print('DATASET loaded')

    # create LEAVE ONE OUT (LOO) cross-validation
    LOO = LeaveOneOut()
    # create day column
    df_har['day'] = pd.DatetimeIndex(df_har.index).normalize().astype(str)
    # group by day and filter days with less than a minimum number of sensor events
    df_har = df_har.groupby(by=["day"]).filter(lambda day: len(day) >= args.min_sensor_events)
    print("Total number of sensor events after filtering: " + str(len(df_har)))
    # get unique days for cross-validation
    days = set(df_har['day'])
    sorted_days = []
    for day in days:
        sorted_days.append(day)
    sorted_days = sorted(sorted_days)
    # get indexes of LEAVE ONE DAY OUT (LODO) cross-validation
    train_test_indexes = LOO.split(sorted_days)

    print('Starting LEAVE ONE DAY OUT cross-validation to generate models...')
    for train_indexes, test_index in train_test_indexes:
        # get train days data
        train_days = []
        for train_index in train_indexes:
            train_days.append(sorted_days[train_index])
        df_har_train = df_har[df_har['day'].isin(train_days)]
        # get test day
        test_day = sorted_days[test_index[0]]
        # train word2vec model
        model = Word2Vec([df_har_train['action'].values.tolist()],
                size=args.embedding_size, window=args.window_size, min_count=0, iter=args.iterations, seed=np.random.randint(1000000),
                workers=multiprocessing.cpu_count())
        # save word2vec model
        model.save(args.results_dir + "/" + args.dataset_dir.split('/')[2] + "/" 'word2vec_models/' + str(test_day) + '_day.model')
        model.wv.save_word2vec_format(args.results_dir + "/" + args.dataset_dir.split('/')[2] + "/" 'word2vec_models/vector_files/' + str(test_day) + '_day.vector', binary=False)
    print('Finished LEAVE ONE DAY OUT cross-validation to generate models...')

if __name__ == "__main__":
    main(sys.argv)