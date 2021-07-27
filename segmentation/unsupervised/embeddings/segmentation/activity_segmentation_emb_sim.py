import json
import sys
import numpy as np
import pandas as pd
import argparse

from sklearn import preprocessing
from pylab import *
from math import sqrt

sys.path.insert(1, '/activity_segmentation/preprocessing')
sys.path.insert(1, '/activity_segmentation/segmentation/evaluate/utils')
sys.path.insert(1, '/activity_segmentation/utils')

from activity_datasets_preprocessing import *
from activity_change_save_results import *
from activity_change_evaluation import *

from create_embedding_matrix import *
from scipy import spatial

from get_change_points import *

import multiprocessing
from gensim.models import Word2Vec

def main(argv):
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(suppress=True)
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
                        default='activity_segmentation/results/kasteren_house_a/reduced',
                        nargs="?",
                        help="Dir for results")
    parser.add_argument("--results_folder",
                        type=str,
                        default='word2vec',
                        nargs="?",
                        help="Folder for results")
    parser.add_argument("--train_or_test",
                        type=str,
                        default='test',
                        nargs="?",
                        help="Specify train or test data")
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
    parser.add_argument("--context_window_size",
                        type=int,
                        default=1,
                        nargs="?",
                        help="Context window size for CPD algorithm")
    parser.add_argument("--iterations",
                        type=int,
                        default=5,
                        nargs="?",
                        help="Iterations for word2vec algorithm")
    parser.add_argument("--threshold",
                        type=float,
                        default=0.5,
                        nargs="?",
                        help="Threshold to perform score strategy CPD")
    parser.add_argument("--exe",
                        type=int,
                        default=5,
                        nargs="?",
                        help="Number of executions")
    args = parser.parse_args()
    print('Loading dataset...')
    sys.stdout.flush()
    # dataset of actions and activities
    DATASET = args.dataset_dir + "/" + args.dataset_file.replace('.csv', '') + "_" + args.train_or_test + ".csv"
    df_dataset = pd.read_csv(DATASET, parse_dates=[[0, 1]], header=None, index_col=0, sep=' ')
    df_dataset.columns = ['sensor', 'action', 'event', 'activity']
    df_dataset.index.names = ["timestamp"]
    # list of unique actions in the dataset
    UNIQUE_ACTIONS = args.dataset_dir + "/" + 'unique_actions.json'
    # context information for the actions in the dataset
    CONTEXT_OF_ACTIONS = args.dataset_dir + "/" + 'context_model.json'
    # total actions and its names
    unique_actions = json.load(open(UNIQUE_ACTIONS, 'r'))
    total_actions = len(unique_actions)
    # context of actions
    context_of_actions = json.load(open(CONTEXT_OF_ACTIONS, 'r'))
    action_location = {}
    for key, values in context_of_actions['objects'].items():
        action_location[key] = values['location']
    # check action:location dict struct
    print(action_location)
    # check dataset struct
    print("Dataset")
    print(df_dataset)
    # prepare dataset
    X, timestamps, days, hours, seconds_past_midnight, y, tokenizer_action = prepare_x_y_activity_change(df_dataset)
    # transform action:location dict struct to action_index:location struct
    action_index = tokenizer_action.word_index
    action_index_location = {}
    for key, value in action_index.items():
        action_index_location[value] = action_location[key]
    # check action_index:location struct
    print(action_index_location)
    # check prepared dataset struct
    print("Actions")
    print(X)
    print("Activity change")
    print(y)
    # change point detection
    window_size = args.window_size
    context_window_size = args.context_window_size
    iterations = args.iterations
    embedding_size = args.embedding_size
    exe = args.exe
    # check actions input shape
    print("Input action shape: " + str(X.shape))
    for e in range(0, exe):
        RESULTS_DIR = "/" + args.results_dir + "/" + args.results_folder + "/window_" + str(window_size) + "_iterations_" + str(iterations) + "_embedding_size_" + str(embedding_size) + "/" + args.train_or_test + "/"
        # create embedding matrix from word2vec retrofitted vector file
        model = Word2Vec.load(RESULTS_DIR.replace("test", "train") + 'word2vec_models/' + str(e) + '_execution.model')
        embedding_action_matrix, unknown_actions = create_action_embedding_matrix(tokenizer_action, model, embedding_size)
        # calculate similarities using word2vec embeddings
        similarities = []
        similarities.append(1.0)
        for i in range(0, len(X)-1):
            similarity = 1 - spatial.distance.cosine(embedding_action_matrix[X[i]], embedding_action_matrix[X[i+1]])
            similarities.append(similarity)
        # prepare change detection segmentation
        threshold = args.threshold
        change_points = get_change_points_lower(similarities, threshold)
        label = 'Activity'
        label_number = 0
        labels = []
        for i in range(0, len(X)):
            if change_points[i] == 0:
                labels.append(label + ' ' + str(label_number))
            else:
                label_number += 1
                labels.append(label + ' ' + str(label_number))
        df_dataset['Predicted_Segment'] = labels
        # write to file
        df_dataset.to_csv(RESULTS_DIR + "segmentation_" + str(e) + ".csv")
    # mark experiment end
    print('... Experiment finished ...')
    print('Results saved to: ' + RESULTS_DIR)
    print('... ... ... ... ... ... ...')

if __name__ == "__main__":
    main(sys.argv)