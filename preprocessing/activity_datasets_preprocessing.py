import numpy as np
import pandas as pd
import itertools
import operator

from keras.preprocessing.text import Tokenizer
from datetime import datetime as dt

def get_unixtime(dt64):
    return dt64.astype('datetime64[s]').astype('int')

def convert_epoch_time_to_hour_of_day(epoch_time_in_seconds):
    d = dt.fromtimestamp(epoch_time_in_seconds)
    return d.strftime('%H')

def convert_epoch_time_to_day_of_the_week(epoch_time_in_seconds):
    d = dt.fromtimestamp(epoch_time_in_seconds)
    return d.strftime('%A')

def get_seconds_past_midnight_from_epoch(epoch_time_in_seconds):
    date = dt.fromtimestamp(epoch_time_in_seconds)
    seconds_past_midnight = (date - date.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    return seconds_past_midnight

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

def longest_segment_length(segments):
    max_segment_length = 0
    for segment in segments:
        if len(segment) > max_segment_length:
            max_segment_length = len(segment)
    return max_segment_length

# https://stackoverflow.com/questions/752308/split-list-into-smaller-lists-split-in-half
def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

def prepare_x_y_activity_change(df):
    actions = df['action'].values
    dates = list(df.index.values)
    timestamps = list(map(get_unixtime, dates))
    hours = list(map(convert_epoch_time_to_hour_of_day, timestamps))
    days = list(map(convert_epoch_time_to_day_of_the_week, timestamps))
    seconds_past_midnight = list(map(get_seconds_past_midnight_from_epoch, timestamps))
    activities = df['activity'].values

    tokenizer_actions = Tokenizer(lower=False)
    tokenizer_actions.fit_on_texts(actions.tolist())
    action_index = tokenizer_actions.word_index
    
    actions_by_index = []
    for action in actions:
        actions_by_index.append(action_index[action])
    
    X = []
    y = []
    last_activity = None
    for i in range(0, len(actions)):
        X.append(actions_by_index[i])
        if (i == 0):
            y.append(0)
        elif last_activity == activities[i]:
            y.append(0)
        else:
            y.append(1)
        last_activity = activities[i]
    
    X = np.array(X)
    timestamps = np.array(timestamps)
    days = np.array(days)
    hours = np.array(hours)
    seconds_past_midnight = np.array(seconds_past_midnight)
    y = np.array(y)
    
    return X, timestamps, days, hours, seconds_past_midnight, y, tokenizer_actions
