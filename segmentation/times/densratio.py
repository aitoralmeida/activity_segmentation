import sys
import numpy as np
import time
import random

sys.path.append('/activity_segmentation/segmentation/unsupervised/likelihood/densratio')
from core import densratio

def main(argv):

    for FEATURES in [15, 30, 50, 75, 100, 250, 500]:
        vector_1 = []
        vector_2 = []
        for i in range(0, FEATURES):
            vector_1.append(random.uniform(0, 1))
            vector_2.append(random.uniform(0, 1))
        vector_1 = np.array(vector_1)
        vector_2 = np.array(vector_2)
        start_time = time.time()
        densratio_obj = densratio(vector_1, vector_2, alpha=0.01)
        end_time = time.time()
        used_time = end_time - start_time
        print("TIME for vectors with NUMBER OF FEATURES " + str(FEATURES) + ": " + str(format(used_time, 'f')))

if __name__ == "__main__":
    main(sys.argv)