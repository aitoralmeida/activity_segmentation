import json
import sys
import argparse
import time
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

import wandb
from wandb.keras import WandbCallback

from models import LSTM_EMBEDDING_MODEL

# A3 ARCHITECTURE (LSTM)
 
BEST_MODEL = 'best_model.hdf5'
def select_best_model():
    model = load_model(BEST_MODEL)
    return model
         
def prepare_x_y(df, unique_actions, number_of_actions):
    # recover all the actions in order
    actions = df['action'].values
    timestamps = df.index.tolist()
    print(('total actions', len(actions)))
    # use tokenizer to generate indices for every action
    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(actions.tolist())
    action_index = tokenizer.word_index  
    # translate actions to indexes
    actions_by_index = []
    for action in actions:
        actions_by_index.append(action_index[action])
    # create the training sets of sequences with a lenght of number_of_actions
    last_action = len(actions) - 1
    X_actions = []
    y = []
    for i in range(last_action-number_of_actions):
        X_actions.append(actions_by_index[i:i+number_of_actions])
        # represent the target action as a one-hot for the softmax
        target_action = ''.join(i for i in actions[i+number_of_actions] if not i.isdigit()) # remove the period if it exists
        target_action_onehot = np.zeros(len(unique_actions))
        target_action_onehot[unique_actions.index(target_action)] = 1.0
        y.append(target_action_onehot)
    return X_actions, y, tokenizer
    
def create_action_embedding_matrix_from_file(tokenizer, vector_file, embedding_size):
    data = pd.read_csv(vector_file, sep=",", header=None)
    data.columns = ["action", "vector"]
    action_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(action_index) + 1, embedding_size))
    unknown_actions = {}    
    for action, i in list(action_index.items()):
        try:
            embedding_vector = np.fromstring(data[data['action'] == action]['vector'].values[0], dtype=float, sep=' ')
            embedding_matrix[i] = embedding_vector        
        except:
            if action in unknown_actions:
                unknown_actions[action] += 1
            else:
                unknown_actions[action] = 1
    
    return embedding_matrix, unknown_actions

def main(argv):
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(suppress=True)
    pd.options.mode.chained_assignment = None
    # parse args
    parser = argparse.ArgumentParser()
    # general args
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
    parser.add_argument("--number_of_actions",
                        type=int,
                        default=5,
                        nargs="?",
                        help="Number of actions to input to the LSTM")
    parser.add_argument("--optimizer",
                        type=str,
                        default='adam',
                        nargs="?",
                        help="Optimizer type")
    # word2vec parameters
    parser.add_argument("--retrofitted_embeddings",
                        type=str,
                        default='False',
                        nargs="?",
                        help="If retrofitted embeddings must be used")
    parser.add_argument("--graph_to_retrofit",
                        type=str,
                        default='activities',
                        nargs="?",
                        help="Graph if retrofitted_embedding is True")
    parser.add_argument("--word2vec_window",
                        type=int,
                        default=5,
                        nargs="?",
                        help="Word2vec window")
    parser.add_argument("--iterations",
                        type=int,
                        default=5,
                        nargs="?",
                        help="Word2vec iterations")
    parser.add_argument("--embedding_size",
                        type=int,
                        default=50,
                        nargs="?",
                        help="Word2vec embedding size")
    parser.add_argument("--trainable_embeddings",
                        type=str,
                        default='True',
                        nargs="?",
                        help="Word2vec embeddings trainable in model or not (True or False)")
    args = parser.parse_args()
    
    sys.stdout.flush()

    # init wandb
    run = wandb.init(project='har_futuraal', config=args, tags=['LSTM_NEXT_ACTION_PREDICTION'])
    config = wandb.config

    # loading full dataset
    print('Loading DATASET...')
    DATASET = args.dataset_dir + "/" + args.dataset_file
    df_har = pd.read_csv(DATASET, parse_dates=[[0, 1]], index_col=0, sep=' ', header=None)
    df_har.columns = ['sensor', 'action', 'event', 'activity']
    df_har.index.names = ["timestamp"]
    print('DATASET loaded')
    
    # we only need the unique actions to calculate the one-hot vector for y, because we are only predicting the actions
    UNIQUE_ACTIONS = args.dataset_dir + "/" + 'unique_actions.json'
    unique_actions = json.load(open(UNIQUE_ACTIONS, 'r'))
    total_actions = len(unique_actions)
    
    print(('*' * 20))
    print('Preparing dataset...')
    sys.stdout.flush()
    # prepare sequences using action indices
    # each action will be an index which will point to an action vector
    # in the weights matrix of the embedding layer of the network input
    X_actions, y, tokenizer = prepare_x_y(df_har, unique_actions, args.number_of_actions)    

    # create the embedding matrix for the embedding layer initialization from FILE
    if args.retrofitted_embeddings == 'True':
        VECTOR_FILE = args.dataset_dir + '/word2vec_models/word2vec_retrofitted_' + str(args.graph_to_retrofit) + '_embedding_size_' + str(args.embedding_size) + '_iterations_' + str(args.iterations) + '_word2vec_window_' + str(args.word2vec_window)
    else:
        VECTOR_FILE = args.dataset_dir + '/word2vec_models/word2vec_embedding_size_' + str(args.embedding_size) + '_iterations_' + str(args.iterations) + '_word2vec_window_' + str(args.word2vec_window)
    embedding_matrix, unknown_actions = create_action_embedding_matrix_from_file(tokenizer, VECTOR_FILE, args.embedding_size)
    
    # divide the examples in training and validation
    total_examples = len(X_actions)
    test_per = 0.2
    limit = int(test_per * total_examples)
    X_actions_train = X_actions[limit:]
    X_actions_test = X_actions[:limit]
    y_train = y[limit:]
    y_test = y[:limit]
    print(('Different actions:', total_actions))
    print(('Total examples:', total_examples))
    print(('Train examples:', len(X_actions_train), len(y_train))) 
    print(('Test examples:', len(X_actions_test), len(y_test)))
    sys.stdout.flush()  
    X_actions_train = np.array(X_actions_train)
    y_train = np.array(y_train)
    X_actions_test = np.array(X_actions_test)
    y_test = np.array(y_test)
    print('Shape (X,y):')
    print((X_actions_train.shape))
    print((y_train.shape))
   
    print(('*' * 20))
    print('Building model...')
    sys.stdout.flush()
    # build model to predict next action
    trainable_embeddings = True if args.trainable_embeddings == 'True' else False
    model = LSTM_EMBEDDING_MODEL(args.number_of_actions, len(unique_actions), embedding_matrix, args.embedding_size, trainable_embeddings)
    
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer=args.optimizer, metrics=['accuracy', 'mse', 'mae'])
    
    # show model summary
    model.summary()
    sys.stdout.flush()
    
    print(('*' * 20))
    print('Training model...')    
    sys.stdout.flush()
    # train model
    checkpoint = ModelCheckpoint(BEST_MODEL, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    model.fit(X_actions_train, y_train, batch_size=128, epochs=1000, validation_data=(X_actions_test, y_test), shuffle=False, callbacks=[checkpoint, WandbCallback()])

    print(('*' * 20))
    print('Evaluating best model...')
    sys.stdout.flush()
    # evaluate best model with accuracy at 5
    model = select_best_model()
    predictions = model.predict(X_actions_test, 128)
    correct = [0] * 5
    prediction_range = 5
    for i, prediction in enumerate(predictions):
        correct_answer = y_test[i].tolist().index(1)       
        best_n = np.sort(prediction)[::-1][:prediction_range]
        for j in range(prediction_range):
            if prediction.tolist().index(best_n[j]) == correct_answer:
                for k in range(j,prediction_range):
                    correct[k] += 1 
    
    # log accurracy at 5                 
    for i in range(prediction_range):
        wandb.log({'acc_at_' + str(i+1): (correct[i] * 1.0) / len(y_test)})

    # clean keras session
    tf.keras.backend.clear_session()
    
    print(('************ END ************\n' * 3))   


if __name__ == "__main__":
    main(sys.argv)