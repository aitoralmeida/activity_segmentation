import sys
import numpy as np

sys.path.append('../densratio')
from core import densratio

def change_detection(feature_vectors, n, alpha):
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

        # Found optimal values for sigma = [10.000, 100.000, 1000.000], lambda = [0.010, 0.0010] through empirical leave-one-out CV.
        densratio_obj = densratio(y_test, y_ref, alpha=alpha, sigma_range=[10.000, 100.000, 1000.000], lambda_range=[0.010, 0.0010])

        scores.append(densratio_obj.alpha_SEP)

        t += 1
    
    print("Num of scores: " + str(len(scores)))

    return scores
