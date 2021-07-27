import json
import sys
import numpy as np
import pandas as pd
import argparse
import itertools
import operator
import csv

sys.path.insert(1, '/activity_segmentation/preprocessing')
sys.path.insert(1, '/activity_segmentation/segmentation/evaluate/utils')

from activity_datasets_preprocessing import *
from activity_change_save_results import *

from sklearn import preprocessing
from pylab import *
from scipy.stats import entropy

##################################################################################################################
# Feature extraction approach of Real-Time Change Point Detection with Application to Smart Home Time Series Data
# https://ieeexplore.ieee.org/document/8395405
# START
##################################################################################################################

def sliding_window_with_features(actions, unique_actions, locations, timestamps, days, hours, seconds_past_midnight, k, norm='True', type_norm='min_max'):
    feature_vectors = []
    previous_actions_1 = None
    previous_actions_2 = None
    num_samples = len(actions)

    for i in range(0, num_samples):
        offset = k
        if i + offset > num_samples:
            break
        window_actions = actions[i:(i+offset)]
        window_timestamps = timestamps[i:(i+offset)]
        # get feature vector from window
        feature_vector = []
        # time features
        feature_vector.append(int(hours[i+offset-1]))
        feature_vector.append(day_to_int(days[i+offset-1]))
        feature_vector.append(seconds_past_midnight[i+offset-1])
        # window features
        window_features = extract_features_from_window(window_actions, previous_actions_1, previous_actions_2, locations, window_timestamps)
        feature_vector.extend(window_features)
        # sensor features
        sensor_features = extract_features_from_sensors(window_actions, unique_actions, actions, i+offset-1, timestamps)
        feature_vector.extend(sensor_features)
        # update previous actions
        previous_actions_2 = previous_actions_1
        previous_actions_1 = window_actions
        # add to windows struct
        feature_vectors.append(np.array(feature_vector))
    
    feature_vectors = np.array(feature_vectors)
    if norm == 'True':
        if type_norm == 'min_max':
            feature_vectors = (feature_vectors - feature_vectors.min(axis=0)) / (feature_vectors.max(axis=0) - feature_vectors.min(axis=0))
        elif type_norm == 'z_score':
            feature_vectors = (feature_vectors - feature_vectors.mean(axis=0)) / feature_vectors.std(axis=0)

    return feature_vectors

def most_common(L):
    SL = sorted((x, i) for i, x in enumerate(L))

    groups = itertools.groupby(SL, key=operator.itemgetter(0))

    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        return count, -min_index

    return max(groups, key=_auxfun)[0]

def calc_entropy(labels, base=None):
    value,counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)

def day_to_int(day):
    switcher = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    return switcher[day]

def location_to_int(location):
    switcher = {"Bathroom": 0, "Kitchen": 1, "Bedroom": 2, "Hall": 3, "Office": 4, "LivingRoom": 5, "BathroomDownstairs": 6, "BathroomUpstairs": 7, "Balcony": 8}
    return switcher[location]

def extract_actions_from_window(actions, timestamps, position, k=30):
    actions_from_window = []
    timestamps_from_window = []

    for i in range(position-(k), position):
        if i >= 0:
            actions_from_window.append(actions[i])
            timestamps_from_window.append(timestamps[i])
    actions_from_window.append(actions[position])
    timestamps_from_window.append(timestamps[position])

    return actions_from_window, timestamps_from_window

def extract_features_from_window(actions, previous_actions_1, previous_actions_2, locations, timestamps):
    features_from_window = []
    
    most_recent_sensor = actions[-1]
    first_sensor_in_window = actions[0]
    window_duration = timestamps[-1] - timestamps[0]
    if previous_actions_1 is not None:
        most_frequent_sensor_1 = most_common(previous_actions_1)
    else:
        most_frequent_sensor_1 = 0 # non-existing sensor index
    if previous_actions_2 is not None:
        most_frequent_sensor_2 = most_common(previous_actions_2)
    else:
        most_frequent_sensor_2 = 0 # non-existing sensor index
    last_sensor_location = location_to_int(locations[actions[-1]])
    # last_motion_sensor_location -> no aplica en kasteren !!!!!
    entropy_based_data_complexity = calc_entropy(actions)
    time_elapsed_since_last_sensor_event = timestamps[-1] - timestamps[-2]

    features_from_window.append(most_recent_sensor)
    features_from_window.append(first_sensor_in_window)
    features_from_window.append(window_duration)
    features_from_window.append(most_frequent_sensor_1)
    features_from_window.append(most_frequent_sensor_2)
    features_from_window.append(last_sensor_location)
    features_from_window.append(entropy_based_data_complexity)
    features_from_window.append(time_elapsed_since_last_sensor_event)

    return features_from_window

def extract_features_from_sensors(actions, unique_actions, all_actions, position, all_timestamps):
    features_from_sensors = []
    
    # count of events for each sensor in window
    for action in unique_actions:
        counter = 0
        for action_fired in actions:
            if action == action_fired:
                counter += 1
        features_from_sensors.append(counter)
    
    # elapsed time for each sensor since last event
    found_actions = []
    counter = position
    last_event_timestamp = all_timestamps[position]
    last_sensor_timestamp = None
    for action in unique_actions:
        while(counter >= 0):
            if action == all_actions[counter]:
                found_actions.append(action)
                last_sensor_timestamp = all_timestamps[counter]
                break
            counter -= 1
        if action not in found_actions:
            features_from_sensors.append(all_timestamps[-1] - all_timestamps[0]) # maximum time possible
        else:
            features_from_sensors.append(last_event_timestamp - last_sensor_timestamp)
        counter = position
        last_sensor_timestamp = None

    return features_from_sensors

##################################################################################################################
# Feature extraction approach of Real-Time Change Point Detection with Application to Smart Home Time Series Data
# https://ieeexplore.ieee.org/document/8395405
# END
##################################################################################################################

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
                        default='feature_extraction',
                        nargs="?",
                        help="Folder for results")
    parser.add_argument("--train_or_test",
                        type=str,
                        default='train',
                        nargs="?",
                        help="Specify train or test data")
    parser.add_argument("--k",
                        type=int,
                        default=30,
                        nargs="?",
                        help="Number of events to look when performing feature extraction")
    parser.add_argument("--norm",
                        type=str,
                        default='False',
                        nargs="?",
                        help="Feature vector normalization")
    parser.add_argument("--type_norm",
                        type=str,
                        default='min_max',
                        nargs="?",
                        help="Feature vector normalization strategy")
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
    # feature extraction
    k = args.k
    norm = args.norm
    type_norm = args.type_norm
    feature_vectors = sliding_window_with_features(X, action_index.values(), action_index_location, timestamps, days, hours, seconds_past_midnight, k, norm, type_norm)
    # save to file
    RESULTS_DIR = "/" + args.results_dir + "/" + args.results_folder + "/" + args.train_or_test + "/"
    create_dirs(RESULTS_DIR, word2vec=False)
    if norm == 'True':
        OUTPUT_FILE = RESULTS_DIR + 'k_' + str(k) + '_feature_vectors_norm_' + type_norm + '.csv'
    else:
        OUTPUT_FILE = RESULTS_DIR + 'k_' + str(k) + '_feature_vectors.csv'
    def vector_to_str(feature_vector):
        return ','.join(['%.10f' % num for num in feature_vector])
    features_vectors_str = list(map(vector_to_str, feature_vectors))
    with open(OUTPUT_FILE, "w") as f:
        for feature_vector_str in features_vectors_str:
            f.write(feature_vector_str + "\n")
    # mark feature extraction end
    print('... Feature extraction finished ...')
    print('Results saved to: ' + OUTPUT_FILE)
    print('... ... ... ... ... ... ...')

if __name__ == "__main__":
    main(sys.argv)