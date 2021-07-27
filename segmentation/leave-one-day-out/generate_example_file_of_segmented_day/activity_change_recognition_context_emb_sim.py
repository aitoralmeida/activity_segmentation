import json
import sys
import numpy as np
import pandas as pd
import argparse

from sklearn import preprocessing
from scipy import spatial
from math import sqrt
from datetime import datetime as dt

sys.path.insert(1, '/activity_segmentation/segmentation/evaluate/utils')
sys.path.insert(1, '/activity_segmentation/utils')

from activity_change_evaluation import *

from create_embedding_matrix import *
from calculate_context_similarity import *

import multiprocessing
from gensim.models import Word2Vec

from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import LeaveOneOut

def get_unixtime(dt64):
    return dt64.astype('datetime64[s]').astype('int')

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
    parser.add_argument("--models_dir",
                        type=str,
                        default="../generated_models",
                        nargs="?",
                        help="Dataset dir")
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
    parser.add_argument("--graph_to_retrofit",
                        type=str,
                        default='None',
                        nargs="?",
                        help="Graph to retrofit")
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

    # load contextual and action information
    UNIQUE_ACTIONS = args.dataset_dir + "/" + 'unique_actions.json'
    CONTEXT_OF_ACTIONS = args.dataset_dir + "/" + 'context_model.json'
    unique_actions = json.load(open(UNIQUE_ACTIONS, 'r'))
    total_actions = len(unique_actions)
    # context of actions
    context_of_actions = json.load(open(CONTEXT_OF_ACTIONS, 'r'))
    action_location = {}
    for key, values in context_of_actions['objects'].items():
        action_location[key] = values['location']

    # tokenize actions
    tokenizer_actions = Tokenizer(lower=False)
    tokenizer_actions.fit_on_texts(df_har['action'])
    action_index = tokenizer_actions.word_index

    # tokenize activities
    tokenizer_activities = Tokenizer(lower=False)
    tokenizer_activities.fit_on_texts(df_har['activity'])
    activity_index = tokenizer_activities.word_index

    # prepare mean metrics for exact
    TPR_list_1 = []
    TNR_list_1 = []
    FPR_list_1 = []
    G_MEAN_list_1 = []
     # prepare mean metrics for 5s offset
    TPR_list_5 = []
    TNR_list_5 = []
    FPR_list_5 = []
    G_MEAN_list_5 = []
     # prepare mean metrics for 10s offset
    TPR_list_10 = []
    TNR_list_10 = []
    FPR_list_10 = []
    G_MEAN_list_10 = []

    # iterate through LODO cross-validation indexes
    print('Starting LEAVE ONE DAY OUT cross-validation...')
    for train_indexes, test_index in train_test_indexes:
        
        # get train days data
        train_days = []
        for train_index in train_indexes:
            train_days.append(sorted_days[train_index])
        df_har_train = df_har[df_har['day'].isin(train_days)]
        print("Total number of train sensor events: " + str(len(df_har_train)))
        # create actions and activities by index train
        actions_by_index_train = []
        for action in df_har_train['action']:
            actions_by_index_train.append(action_index[action])
        activities_by_index_train = []
        for activity in df_har_train['activity']:
            activities_by_index_train.append(activity_index[activity])
        # take timestamps train in seconds
        dates_train = list(df_har_train.index.values)
        timestamps_train = list(map(get_unixtime, dates_train))
        # generate X (actions) and y (change points) for train
        X_train = []
        y_train = []
        last_activity = None
        for i in range(0, len(actions_by_index_train)):
            X_train.append(actions_by_index_train[i])
            if (i == 0):
                y_train.append(0)
            elif last_activity == activities_by_index_train[i]:
                y_train.append(0)
            else:
                y_train.append(1)
            last_activity = activities_by_index_train[i]
        print("Total number of train samples: " + str(len(X_train)))
        print("Total number of train labels: " + str(len(y_train)))

        # get test day data
        test_day = sorted_days[test_index[0]]
        df_har_test = df_har[df_har['day'] == test_day]
        print("Total number of test sensor events: " + str(len(df_har_test)))
        # create actions and activities by index test
        actions_by_index_test = []
        for action in df_har_test['action']:
            actions_by_index_test.append(action_index[action])
        activities_by_index_test = []
        for activity in df_har_test['activity']:
            activities_by_index_test.append(activity_index[activity])
        # take timestamps test in seconds
        dates_test = list(df_har_test.index.values)
        timestamps_test = list(map(get_unixtime, dates_test))
        # generate X (actions) and y (change points) for test
        X_test = []
        y_test = []
        last_activity = None
        for i in range(0, len(actions_by_index_test)):
            X_test.append(actions_by_index_test[i])
            if (i == 0):
                y_test.append(0)
            elif last_activity == activities_by_index_test[i]:
                y_test.append(0)
            else:
                y_test.append(1)
            last_activity = activities_by_index_test[i]
        print("Total number of test samples: " + str(len(X_test)))
        print("Total number of test labels: " + str(len(y_test)))

        # load word2vec model and create embedding matrix
        if args.graph_to_retrofit == 'None':
            model = Word2Vec.load(args.models_dir + "/" + args.dataset_dir.split('/')[2] + "/" 'word2vec_models/' + str(test_day) + '_day.model')
            embedding_action_matrix, unknown_actions = create_action_embedding_matrix(tokenizer_actions, model, args.embedding_size)
        else:
            vector_file = args.models_dir + "/" + args.dataset_dir.split('/')[2] + "/" 'word2vec_models/vector_files/' + str(test_day) + '_retrofitted_' + args.graph_to_retrofit + '.vector'
            embedding_action_matrix, unknown_actions = create_action_embedding_matrix_from_file(tokenizer_actions, vector_file, args.embedding_size)

        # calculate optimum threshold with training data
        best_threshold_global = 0
        best_G_MEAN_global = 0
        best_context_window_size = 1
        for context_window_size in [1, 2, 3, 4, 5]:
            similarities_train = []
            similarities_train.append(0)
            for i in range(1, len(X_train)):
                context_similarity = calculate_context_similarity(X_train, embedding_action_matrix, i, context_window_size)
                similarities_train.append(context_similarity)
            best_threshold = 0
            best_G_MEAN = 0
            for threshold in [x * 0.1 for x in range(0, 10)]:
                cf_matrix_1 = get_conf_matrix_with_offset_strategy(similarities_train, y_train, timestamps_train, threshold, 1)
                # TPR, TNR, FPR, G-MEAN for exact change point detection
                TN, FP, FN, TP = cf_matrix_1.ravel()
                TPR = TP/(TP+FN)
                TNR = TN/(TN+FP)
                FPR = FP/(FP+TN)
                G_MEAN = sqrt(TPR * TNR)
                if G_MEAN > best_G_MEAN:
                    best_G_MEAN = G_MEAN
                    best_threshold = threshold
            if best_G_MEAN > best_G_MEAN_global:
                best_G_MEAN_global = best_G_MEAN
                best_threshold_global = best_threshold
                best_context_window_size = context_window_size
        print("-----------------------------------------------------------------")
        print("DAY: " + str(test_day))
        print("Calculated optimum context window size: " + str(best_context_window_size))
        print("Calculated optimum threshold: " + str(best_threshold_global))
        print("Best G-MEAN value on TRAIN data (EXACT): " + str(best_G_MEAN_global))

        # calculate metrics in test data
        similarities_test = []
        similarities_test.append(0)
        for i in range(1, len(X_test)):
            context_similarity = calculate_context_similarity(X_test, embedding_action_matrix, i, best_context_window_size)
            similarities_test.append(context_similarity)
        # TPR, TNR, FPR, G-MEAN for exact change point detection
        cf_matrix_1 = get_conf_matrix_with_offset_strategy(similarities_test, y_test, timestamps_test, best_threshold_global, 1)
        TN, FP, FN, TP = cf_matrix_1.ravel()
        TPR = TP/(TP+FN)
        TNR = TN/(TN+FP)
        FPR = FP/(FP+TN)
        G_MEAN = sqrt(TPR * TNR)
        TPR_list_1.append(TPR)
        TNR_list_1.append(TNR)
        FPR_list_1.append(FPR)
        G_MEAN_list_1.append(G_MEAN)
        print("-----------------------------------------------------------------")
        print("TPR value on TEST data (EXACT): " + str(TPR))
        print("FPR value on TEST data (EXACT): " + str(FPR))
        print("G-MEAN value on TEST data (EXACT): " + str(G_MEAN))
        # TPR, TNR, FPR, G-MEAN for 5s offset
        cf_matrix_5 = get_conf_matrix_with_offset_strategy(similarities_test, y_test, timestamps_test, best_threshold_global, 5)
        TN, FP, FN, TP = cf_matrix_5.ravel()
        TPR = TP/(TP+FN)
        TNR = TN/(TN+FP)
        FPR = FP/(FP+TN)
        G_MEAN = sqrt(TPR * TNR)
        TPR_list_5.append(TPR)
        TNR_list_5.append(TNR)
        FPR_list_5.append(FPR)
        G_MEAN_list_5.append(G_MEAN)
        print("-----------------------------------------------------------------")
        print("TPR value on TEST data (5s): " + str(TPR))
        print("FPR value on TEST data (5s): " + str(FPR))
        print("G-MEAN value on TEST data (5s): " + str(G_MEAN))
        # TPR, TNR, FPR, G-MEAN for 10s offset
        cf_matrix_10 = get_conf_matrix_with_offset_strategy(similarities_test, y_test, timestamps_test, best_threshold_global, 10)
        TN, FP, FN, TP = cf_matrix_10.ravel()
        TPR = TP/(TP+FN)
        TNR = TN/(TN+FP)
        FPR = FP/(FP+TN)
        G_MEAN = sqrt(TPR * TNR)
        TPR_list_10.append(TPR)
        TNR_list_10.append(TNR)
        FPR_list_10.append(FPR)
        G_MEAN_list_10.append(G_MEAN)
        print("-----------------------------------------------------------------")
        print("TPR value on TEST data (10s): " + str(TPR))
        print("FPR value on TEST data (10s): " + str(FPR))
        print("G-MEAN value on TEST data (10s): " + str(G_MEAN))
        print("-----------------------------------------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------------------------------------")
        segments = []
        label = "Segment_"
        counter = 0
        for similarity in similarities_test:
            if similarity > best_threshold_global:
                counter += 1
            segments.append(label + str(counter))
        df_har_test_with_segments = df_har_test
        df_har_test_with_segments['Segments'] = segments
        df_har_test_with_segments['scores'] = similarities_test
        df_har_test_with_segments[['action', 'activity', 'Segments', 'scores']].to_csv("files/segmentation_" + str(test_day))
        print("SEGMENTS OF DAY " + str(test_day) + " written to file...")
        print("-----------------------------------------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------------------------------------")


    print('LEAVE ONE DAY OUT cross-validation FINISHED')

    print("------ Exact change point detection results ------")
    print("TPR MEAN: " + str(np.mean(TPR_list_1)))
    print("TPR STD: " + str(np.std(TPR_list_1)))
    print("FPR MEAN: " + str(np.mean(FPR_list_1)))
    print("FPR STD: " + str(np.std(FPR_list_1)))
    print("G-Mean MEAN: " + str(np.mean(G_MEAN_list_1)))
    print("G-Mean STD: " + str(np.std(G_MEAN_list_1)))

    print("------ Change point detection with 5s offset results ------")
    print("TPR MEAN: " + str(np.mean(TPR_list_5)))
    print("TPR STD: " + str(np.std(TPR_list_5)))
    print("FPR MEAN: " + str(np.mean(FPR_list_5)))
    print("FPR STD: " + str(np.std(FPR_list_5)))
    print("G-Mean MEAN: " + str(np.mean(G_MEAN_list_5)))
    print("G-Mean STD: " + str(np.std(G_MEAN_list_5)))

    print("------ Change point detection with 10s offset results ------")
    print("TPR MEAN: " + str(np.mean(TPR_list_10)))
    print("TPR STD: " + str(np.std(TPR_list_10)))
    print("FPR MEAN: " + str(np.mean(FPR_list_10)))
    print("FPR STD: " + str(np.std(FPR_list_10)))
    print("G-Mean MEAN: " + str(np.mean(G_MEAN_list_10)))
    print("G-Mean STD: " + str(np.std(G_MEAN_list_10)))

if __name__ == "__main__":
    main(sys.argv)