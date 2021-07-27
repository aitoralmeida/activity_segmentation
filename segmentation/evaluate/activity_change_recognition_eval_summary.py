from pandas import DataFrame
from pandas import read_csv
from matplotlib import pyplot

import sys
import argparse

import numpy as np

def main(argv):
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(suppress=True)
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",
                        type=str,
                        default='/activity_segmentation/results',
                        nargs="?",
                        help="Dir of the results")
    parser.add_argument("--folder",
                        type=str,
                        default='RuLSIF',
                        nargs="?",
                        help="Folder of the results")
    parser.add_argument("--train_or_test",
                        type=str,
                        default='train',
                        nargs="?",
                        help="Specify train or test data results to eval")
    parser.add_argument("--offset",
                        type=int,
                        default=1,
                        nargs="?",
                        help="Offset in seconds")
    parser.add_argument("--threshold_num",
                        type=int,
                        default=10,
                        nargs="?",
                        help="Number of thresholds")
    args = parser.parse_args()

    DIR = args.dir
    FOLDER = args.folder
    OFFSET = args.offset

    for metric in ['TPR', 'TNR', 'FPR', 'G-MEAN']:
        results = DataFrame()
        # add population of results for each threshold
        for threshold in range(0, args.threshold_num):
            results[threshold] = read_csv(DIR + "/" + FOLDER + "/" + args.train_or_test
            + "/" + str(OFFSET) + "s/" + 'results_' + str(OFFSET) + "_" + str(threshold) 
            + "_" + metric + ".csv", header=None).values[:, 0]
        # descriptive stats
        filename = DIR + "/" + FOLDER + "/" + args.train_or_test + "/" + str(OFFSET) + "s/" + 'results_' + str(OFFSET) + "_" + metric
        with open(filename + ".txt", "w") as text_file:
            text_file.write(str(results.describe().apply(lambda s: s.apply(lambda x: format(x, 'g')))))
        # box and whisker plot for each metric
        results.boxplot()
        pyplot.ylim((0.0, 1.0)) 
        pyplot.savefig(filename + ".png")
        pyplot.close()
   
if __name__ == "__main__":
    main(sys.argv)