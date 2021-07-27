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

import networkx as nx
import matplotlib.pyplot as plt

from models import GCN_MODEL_ONLY_EMB
 
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
    # gcn parameters
    parser.add_argument("--activities_graph",
                        type=str,
                        default='/activity_segmentation/segmentation/hybrid/retrofitting/lexicons/kasteren_house_a/reduced/actions_activities.edgelist',
                        nargs="?",
                        help="Graph for adjacency matrix")
    # word2vec parameters
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
    VECTOR_FILE = args.dataset_dir + '/word2vec_models/word2vec_embedding_size_' + str(args.embedding_size) + '_iterations_' + str(args.iterations) + '_word2vec_window_' + str(args.word2vec_window)
    embedding_matrix, unknown_actions = create_action_embedding_matrix_from_file(tokenizer, VECTOR_FILE, args.embedding_size)
    
    # create the model
    model = GCN_MODEL_ONLY_EMB(len(unique_actions), args.embedding_size)
    
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer=args.optimizer, metrics=['accuracy', 'mse', 'mae'])

    # remove padding action 0 and create nodes features with original word2vec embeddings
    embedding_matrix = np.delete(embedding_matrix, 0, 0)
    nodes_features = np.array(embedding_matrix)
    nodes_features = np.expand_dims(nodes_features, axis=0)

    # read graph
    edgelist = pd.read_csv(args.activities_graph, sep=" ", header=None)
    action_index = tokenizer.word_index
    # construct graph
    G = nx.Graph()
    for key, value in action_index.items():
        G.add_node(value)
    for index, row in edgelist.iterrows():
        edge_1 = action_index[row[0]]
        edge_2 = action_index[row[1]]
        G.add_edge(edge_1, edge_2)
    # create adjacency matrix from graph
    A = nx.adjacency_matrix(G).todense()
    A = np.expand_dims(A, axis=0)

    nx.draw_kamada_kawai(G, node_color='r', edge_color='b')
    plt.show(block=False)
    plt.savefig("Graph.png", format="PNG")

    # show model summary
    model.summary()
    sys.stdout.flush()

    # show shapes
    print("Nodes features shape " + str(nodes_features.shape))
    print("Adjacency matrix shape " + str(A.shape))

    # predict model
    model.predict([nodes_features, A])

    # clean keras session
    tf.keras.backend.clear_session()
    
    print(('************ END ************\n' * 3))   


if __name__ == "__main__":
    main(sys.argv)