import sys
import numpy as np
import time
import random

from scipy import spatial

def main(argv):

    for SIZE in [50, 75, 100, 150, 250, 500, 1000]:
        vector_1 = []
        vector_2 = []
        for i in range(0, SIZE):
            vector_1.append(random.uniform(0, 1))
            vector_2.append(random.uniform(0, 1))
        vector_1 = np.array(vector_1)
        vector_2 = np.array(vector_2)
        start_time = time.time()
        similarity = 1 - spatial.distance.cosine(vector_1, vector_2)
        end_time = time.time()
        used_time = end_time - start_time
        print("TIME for vectors of SIZE " + str(SIZE) + ": " + str(format(used_time, 'f')))

if __name__ == "__main__":
    main(sys.argv)