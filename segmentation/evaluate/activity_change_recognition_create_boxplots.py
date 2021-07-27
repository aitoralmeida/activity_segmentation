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
                        default='.',
                        nargs="?",
                        help="Dir of the results")
    parser.add_argument("--folder",
                        type=str,
                        default='boxplots',
                        nargs="?",
                        help="Folder of the results")
    args = parser.parse_args()

    DIR = args.dir
    FOLDER = args.folder

    results = DataFrame()
    for metric in ['uLSIF', 'RuLSIF', 'SEP', 'CS', 'CCS', 'CCSD', 'CCS-R', 'CCSD-R']:
        results[metric] = read_csv(DIR + "/" + FOLDER + "/" + metric + "/" + "results.csv", header=None).values[:, 0]
    results.boxplot(patch_artist=True, 
                    boxprops=dict(facecolor="lightyellow", color="black"),
                    medianprops=dict(color="red"),
                    flierprops=dict(color="black"),
                    meanprops=dict(color="black"),
                    capprops=dict(color="black"),
                    whiskerprops=dict(color="black"),
                    showfliers=False)
    font = {
        'weight': 'bold',
        }
    pyplot.title("CPD with 10s offset", fontdict=font)
    pyplot.ylim((0.0, 1.0))
    pyplot.ylabel("G-Mean")
    pyplot.savefig("/results/boxplots.png")
    pyplot.close() 
   
if __name__ == "__main__":
    main(sys.argv)