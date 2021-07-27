import sys
import numpy as np

sys.path.append('../densratio')
from core import densratio

def change_detection(feature_vectors, n):
    scores = []

    windows = np.zeros((feature_vectors.shape[1], feature_vectors.shape[0]))
    for i in range(0, feature_vectors.shape[0]):
        windows[:,i] = feature_vectors[i,].reshape(1,-1)

    num_samples = windows.shape[1]
    print("Num window samples in change detection: " + str(num_samples))
    t = n

    while((t+n) <= num_samples):
        y = windows[:,(t-n):(n+t)]
        y_ref = y[:,0:n]
        y_test = y[:,n:]

        score = densratio(y_ref, y_test, method="SEP")

        scores.append(score)

        t += 1
    
    print("Num of scores: " + str(len(scores)))

    return scores
