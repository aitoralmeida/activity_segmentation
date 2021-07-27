import numpy as np
import os

def create_dirs(results_dir, word2vec=False):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_dir + "1s", exist_ok=True)
    os.makedirs(results_dir + "5s", exist_ok=True)
    os.makedirs(results_dir + "10s", exist_ok=True)
    if word2vec:
        os.makedirs(results_dir + "word2vec_models", exist_ok=True)

def save_pop_results_to_file(results_dir, results, seconds_for_detection, threshold_num):
    np.savetxt(results_dir + str(seconds_for_detection) + 's/results_' + str(seconds_for_detection) + '_' + str(threshold_num) + '_TPR' + '.csv', results[0][threshold_num])
    np.savetxt(results_dir + str(seconds_for_detection) + 's/results_' + str(seconds_for_detection) + '_' + str(threshold_num) + '_TNR' + '.csv', results[1][threshold_num])
    np.savetxt(results_dir + str(seconds_for_detection) + 's/results_' + str(seconds_for_detection) + '_' + str(threshold_num) + '_FPR' + '.csv', results[2][threshold_num])
    np.savetxt(results_dir + str(seconds_for_detection) + 's/results_' + str(seconds_for_detection) + '_' + str(threshold_num) + '_G-MEAN' + '.csv', results[3][threshold_num])