import numpy as np

################## CPD evaluation confusion matrix ##################
################## ############################## ###################

def get_conf_matrix_with_offset_strategy(scores, y, timestamps, threshold, offset, lower=False):
    cf_matrix = np.zeros((2,2))
    change_points = find_change_points(scores, threshold, lower)
    
    total_detected = sum(change_points)
    total_groundtruth = sum(y)

    TP = 0
    already_used = []

    for i in range(len(y)):
        correct = False
        if y[i] == 1:
            j = i
            while (j > -1) and (timestamps[i] - timestamps[j] < offset):
                j = j - 1
            j = j + 1
            while (correct == False) and (abs(timestamps[j] - timestamps[i]) < offset) and (j < len(y) - 1):
                if change_points[j] == 1:
                    if j not in already_used:
                        correct = True
                        already_used.append(j)
                j = j + 1
            if correct == True:
                TP = TP + 1

    FP = max(0, total_detected - TP)
    FN = max(0, total_groundtruth - TP)
    TN = max(0, len(y) - TP - FP - FN)

    cf_matrix[1][1] = TP
    cf_matrix[0][1] = FP
    cf_matrix[0][0] = TN
    cf_matrix[1][0] = FN
    
    return cf_matrix


def get_conf_matrix_with_offset_strategy_desc(scores, y, timestamps, min_dist, offset):
    cf_matrix = np.zeros((2,2))
    change_points = find_change_points_desc(scores, min_dist)
    
    total_detected = sum(change_points)
    total_groundtruth = sum(y)

    TP = 0
    already_used = []

    for i in range(len(y)):
        correct = False
        if y[i] == 1:
            j = i
            while (j > -1) and (timestamps[i] - timestamps[j] < offset):
                j = j - 1
            j = j + 1
            while (correct == False) and (abs(timestamps[j] - timestamps[i]) < offset) and (j < len(y) - 1):
                if change_points[j] == 1:
                    if j not in already_used:
                        correct = True
                        already_used.append(j)
                j = j + 1
            if correct == True:
                TP = TP + 1

    FP = max(0, total_detected - TP)
    FN = max(0, total_groundtruth - TP)
    TN = max(0, len(y) - TP - FP - FN)

    cf_matrix[1][1] = TP
    cf_matrix[0][1] = FP
    cf_matrix[0][0] = TN
    cf_matrix[1][0] = FN
    
    return cf_matrix


def find_change_points(scores, threshold, lower):
    change_points = []
    for i in range(0, len(scores)):
        if lower:
            if scores[i] < threshold:
                change_points.append(1)
            else:
                change_points.append(0)
        else:
            if scores[i] > threshold:
                change_points.append(1)
            else:
                change_points.append(0)
    return change_points


def find_change_points_desc(scores, min_dist):
    change_points = []
    change_points.append(0)
    for i in range(0, len(scores)-1):
        if scores[i+1] - scores[i] > min_dist:
            change_points.append(1)
        else:
            change_points.append(0)
    return change_points