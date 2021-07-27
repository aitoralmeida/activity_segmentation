from keras.layers import Input, Lambda, Reshape, Dense, Dropout, LSTM, Embedding, Convolution2D, MaxPooling2D, Flatten, Concatenate, Bidirectional, GRU, Multiply, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.models import Model

from keras_gcn import GraphConv, GraphMaxPool

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
        })
        return config

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': self.token_emb,
            'vocab_size': self.pos_emb,
            'embed_dim': self.embed_dim,
        })
        return config

# F2 Transformer architecture with GAPed embedding values
def TRANSFORMER_MODEL_GAP(NUMBER_OF_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS, embedding_size, num_heads, ff_dim, ff_dim_classifier, dropout_rate):
    inputs = layers.Input(shape=(NUMBER_OF_ACTIONS,))
    embedding_layer = TokenAndPositionEmbedding(NUMBER_OF_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS+1, embedding_size)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embedding_size, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(ff_dim_classifier, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(NUMBER_OF_DISTINCT_ACTIONS, activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs)

# F1 Transformer architecture with flattened embedding values
def TRANSFORMER_MODEL_FLATTEN(NUMBER_OF_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS, embedding_size, num_heads, ff_dim, ff_dim_classifier, dropout_rate):
    inputs = layers.Input(shape=(NUMBER_OF_ACTIONS,))
    embedding_layer = TokenAndPositionEmbedding(NUMBER_OF_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS+1, embedding_size)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embedding_size, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(ff_dim_classifier, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(NUMBER_OF_DISTINCT_ACTIONS, activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs)

# A3 architecture in Almeida et al. Predicting Human Behaviour with Recurrent Neural Networks
def LSTM_EMBEDDING_MODEL(NUMBER_OF_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS, embedding_matrix, embedding_size, trainable_embeddings=True):
    # input layer with action sequences sliding window
    actions = Input(shape=(NUMBER_OF_ACTIONS,), name='actions')

    # embedding layer
    embedding_actions = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=NUMBER_OF_ACTIONS, trainable=trainable_embeddings, name='embedding_actions')(actions)    

    # lstm layer
    lstm = LSTM(512, return_sequences=False, input_shape=(NUMBER_OF_ACTIONS, embedding_size), name='lstm')(embedding_actions)
    
    # classification layer fully connected with dropout
    dense_1 = Dense(1024, activation='relu', name='dense_1')(lstm)
    drop_1 = Dropout(0.8, name='drop_1')(dense_1)
    dense_2 = Dense(1024, activation='relu', name='dense_2')(drop_1)
    drop_2 = Dropout(0.8, name='drop_2')(dense_2)
    output_action = Dense(NUMBER_OF_DISTINCT_ACTIONS, activation='softmax', name='output_action')(drop_2)

    return Model(actions, output_action)

# A10 architecture Almeida et al. Predicting Human Behaviour with Recurrent Neural Networks A3 but with ATTENTION
def LSTM_EMBEDDING_LEVEL_ATTENTION(NUMBER_OF_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS, embedding_matrix, embedding_size, trainable_embeddings=True):
    # input layer with action sequences sliding window
    actions = Input(shape=(NUMBER_OF_ACTIONS,), name='actions')

    # embedding layer
    embedding_actions = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=NUMBER_OF_ACTIONS, trainable=trainable_embeddings, name='embedding_actions')(actions)

    # attention layer
    bidirectional_gru = Bidirectional(GRU(embedding_size, input_shape=(NUMBER_OF_ACTIONS, embedding_size), name='bidirectional_gru'))(embedding_actions)  
    dense_att_1 = Dense(embedding_size, activation='tanh', name='dense_att_1')(bidirectional_gru)
    dense_att_2 = Dense(NUMBER_OF_ACTIONS, activation='softmax', name='dense_att_2')(dense_att_1)
    reshape_att = Reshape((NUMBER_OF_ACTIONS, 1), name = 'reshape_att')(dense_att_2)
    apply_att = Multiply()([embedding_actions, reshape_att])

    # lstm layer
    lstm = LSTM(512, return_sequences=False, input_shape=(NUMBER_OF_ACTIONS, embedding_size), name='lstm')(apply_att)
    
    # classification layer fully connected with dropout
    dense_1 = Dense(1024, activation='relu', name='dense_1')(lstm)
    drop_1 = Dropout(0.8, name='drop_1')(dense_1)
    dense_2 = Dense(1024, activation='relu', name='dense_2')(drop_1)
    drop_2 = Dropout(0.8, name='drop_2')(dense_2)
    output_action = Dense(NUMBER_OF_DISTINCT_ACTIONS, activation='softmax', name='output_action')(drop_2)

    return Model(actions, output_action)

# M1 architecture in Almeida et al. Embedding-level attention and multi-scale convolutional neural networks for behaviour modelling
def CNN_EMBEDDING_MODEL(NUMBER_OF_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS, embedding_matrix, embedding_size, trainable_embeddings=True):
    # input layer with action sequences sliding window
    actions = Input(shape=(NUMBER_OF_ACTIONS,), name='actions')

    # embedding layer
    embedding_actions = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=NUMBER_OF_ACTIONS, trainable=trainable_embeddings, name='embedding_actions')(actions)
    reshape = Reshape((NUMBER_OF_ACTIONS, embedding_size, 1), name='reshape')(embedding_actions)

    # convolutional layer
    ngram_2 = Convolution2D(200, (2, embedding_size), padding='valid', activation='relu', name='conv_2')(reshape)
    maxpool_2 = MaxPooling2D(pool_size=(NUMBER_OF_ACTIONS-2+1,1), name='pooling_2')(ngram_2)
    ngram_3 = Convolution2D(200, (3, embedding_size), padding='valid', activation='relu', name='conv_3')(reshape)
    maxpool_3 = MaxPooling2D(pool_size=(NUMBER_OF_ACTIONS-3+1,1), name='pooling_3')(ngram_3)
    ngram_4 = Convolution2D(200, (4, embedding_size), padding='valid', activation='relu', name='conv_4')(reshape)
    maxpool_4 = MaxPooling2D(pool_size=(NUMBER_OF_ACTIONS-4+1,1), name='pooling_4')(ngram_4)
    ngram_5 = Convolution2D(200, (5, embedding_size), padding='valid', activation='relu', name='conv_5')(reshape)
    maxpool_5 = MaxPooling2D(pool_size=(NUMBER_OF_ACTIONS-5+1,1), name='pooling_5')(ngram_5)
    merged = Concatenate(axis=2)([maxpool_2, maxpool_3, maxpool_4, maxpool_5])
    flatten = Flatten(name='flatten')(merged)
    
    # classification layer fully connected with dropout
    dense_1 = Dense(512, activation='relu', name='dense_1')(flatten)
    drop_1 = Dropout(0.8, name='drop_1')(dense_1)
    dense_2 = Dense(512, activation='relu', name='dense_2')(drop_1)
    drop_2 = Dropout(0.8, name='drop_2')(dense_2)
    output_action = Dense(NUMBER_OF_DISTINCT_ACTIONS, activation='softmax', name='output_action')(drop_2)

    return Model(actions, output_action)

# M5 architecture in Almeida et al. Embedding-level attention and multi-scale convolutional neural networks for behaviour modelling
def CNN_EMBEDDING_LEVEL_ATTENTION(NUMBER_OF_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS, embedding_matrix, embedding_size, trainable_embeddings=True):
    # input layer with action sequences sliding window
    actions = Input(shape=(NUMBER_OF_ACTIONS,), name='actions')

    # embedding layer
    embedding_actions = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=NUMBER_OF_ACTIONS, trainable=trainable_embeddings, name='embedding_actions')(actions)    

    # attention layer
    bidirectional_gru = Bidirectional(GRU(embedding_size, input_shape=(NUMBER_OF_ACTIONS, embedding_size), name='bidirectional_gru'))(embedding_actions)  
    dense_att_1 = Dense(embedding_size, activation='tanh', name='dense_att_1')(bidirectional_gru)
    dense_att_2 = Dense(NUMBER_OF_ACTIONS, activation='softmax', name='dense_att_2')(dense_att_1)
    reshape_att = Reshape((NUMBER_OF_ACTIONS, 1), name = 'reshape_att')(dense_att_2)
    apply_att = Multiply()([embedding_actions, reshape_att])
    reshape = Reshape((NUMBER_OF_ACTIONS, embedding_size, 1), name='reshape')(apply_att)

    # convolutional layer
    ngram_2 = Convolution2D(200, (2, embedding_size), padding='valid', activation='relu', name='conv_2')(reshape)
    maxpool_2 = MaxPooling2D(pool_size=(NUMBER_OF_ACTIONS-2+1,1), name='pooling_2')(ngram_2)
    ngram_3 = Convolution2D(200, (3, embedding_size), padding='valid', activation='relu', name='conv_3')(reshape)
    maxpool_3 = MaxPooling2D(pool_size=(NUMBER_OF_ACTIONS-3+1,1), name='pooling_3')(ngram_3)
    ngram_4 = Convolution2D(200, (4, embedding_size), padding='valid', activation='relu', name='conv_4')(reshape)
    maxpool_4 = MaxPooling2D(pool_size=(NUMBER_OF_ACTIONS-4+1,1), name='pooling_4')(ngram_4)
    ngram_5 = Convolution2D(200, (5, embedding_size), padding='valid', activation='relu', name='conv_5')(reshape)
    maxpool_5 = MaxPooling2D(pool_size=(NUMBER_OF_ACTIONS-5+1,1), name='pooling_5')(ngram_5)
    merged = Concatenate(axis=2)([maxpool_2, maxpool_3, maxpool_4, maxpool_5])
    flatten = Flatten(name='flatten')(merged)
    
    # classification layer fully connected with dropout
    dense_1 = Dense(512, activation='relu', name='dense_1')(flatten)
    drop_1 = Dropout(0.8, name='drop_1')(dense_1)
    dense_2 = Dense(512, activation='relu', name='dense_2')(drop_1)
    drop_2 = Dropout(0.8, name='drop_2')(dense_2)
    output_action = Dense(NUMBER_OF_DISTINCT_ACTIONS, activation='softmax', name='output_action')(drop_2)

    return Model(actions, output_action)

# function to take embeddings from graph convolutional layer (must be checked, not sure yet)
def lookup_embeddings(input_tensor):
    actions = input_tensor[0][0]
    actions = actions - 1 # index not action number
    embeddings = input_tensor[1]
    # tf.print(embeddings, summarize=-1)
    picked_embeddings = tf.gather(embeddings, actions, axis=1)
    return picked_embeddings

# GCN architecture (two branches and pick embeddings for the sequence after graph convolution)
def GCN_MODEL(NUMBER_OF_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS, embedding_size, step_num):
    # input layer with action sequences sliding window
    actions = Input(shape=(NUMBER_OF_ACTIONS,), name='actions', dtype=tf.int32)

    # input layer with unique actions (with word2vec embeddings as features) and adjacency matrix
    actions_nodes = Input(shape=(NUMBER_OF_DISTINCT_ACTIONS, embedding_size), name='actions_nodes')
    edges = Input(shape=(NUMBER_OF_DISTINCT_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS), name='edges')

    # graph conv layer which outputs a tensor with shape (NUMBER_OF_DISTINCT_ACTIONS, embedding_size)
    graph_conv = GraphConv(units=embedding_size, step_num=step_num, name='graph_conv')([actions_nodes, edges])

    # pick embeddings for sequence of actions with shape (NUMBER_OF_ACTIONS, embedding_size)
    embeddings_actions = Lambda(lookup_embeddings, output_shape=(NUMBER_OF_ACTIONS, embedding_size), name="embeddings_actions")([actions, graph_conv])
    
    # flatten action embeddings of the sequence
    flatten = Flatten(name='flatten')(embeddings_actions)

    # classification layer fully connected with dropout
    dense_1 = Dense(1024, activation='relu', name='dense_1')(flatten)
    drop_1 = Dropout(0.8, name='drop_1')(dense_1)
    dense_2 = Dense(1024, activation='relu', name='dense_2')(drop_1)
    drop_2 = Dropout(0.8, name='drop_2')(dense_2)
    output_action = Dense(NUMBER_OF_DISTINCT_ACTIONS, activation='softmax', name='output_action')(drop_2)

    return Model([actions, actions_nodes, edges], output_action)

# GCN architecture (2) (not working yet)
def GCN_MODEL_2(NUMBER_OF_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS, embedding_size, step_num):
    # input layer with sequence of actions (with word2vec embeddings as features) and adjacency matrix
    actions_nodes = Input(shape=(NUMBER_OF_DISTINCT_ACTIONS, embedding_size), name='actions_nodes')
    edges = Input(shape=(NUMBER_OF_DISTINCT_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS), name='edges')

    # graph conv layer which outputs a tensor with shape (NUMBER_OF_DISTINCT_ACTIONS, embedding_size)
    graph_conv = GraphConv(units=embedding_size, step_num=step_num, name='graph_conv')([actions_nodes, edges])
    flatten = Flatten(name='flatten')(graph_conv)

    # classification layer fully connected with dropout
    dense_1 = Dense(1024, activation='relu', name='dense_1')(flatten)
    drop_1 = Dropout(0.8, name='drop_1')(dense_1)
    dense_2 = Dense(1024, activation='relu', name='dense_2')(drop_1)
    drop_2 = Dropout(0.8, name='drop_2')(dense_2)
    output_action = Dense(NUMBER_OF_DISTINCT_ACTIONS, activation='softmax', name='output_action')(drop_2)

    return Model([actions_nodes, edges], output_action)

# G1 GCN-LSTM architecture (two branches and pick embeddings for the sequence after graph convolution)
def GCN_LSTM_MODEL(NUMBER_OF_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS, embedding_size, step_num):
    # input layer with action sequences sliding window
    actions = Input(shape=(NUMBER_OF_ACTIONS,), name='actions', dtype=tf.int32)

    # input layer with unique actions (with word2vec embeddings as features) and adjacency matrix
    actions_nodes = Input(shape=(NUMBER_OF_DISTINCT_ACTIONS, embedding_size), name='actions_nodes')
    edges = Input(shape=(NUMBER_OF_DISTINCT_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS), name='edges')

    # graph conv layer which outputs a tensor with shape (NUMBER_OF_DISTINCT_ACTIONS, embedding_size)
    graph_conv = GraphConv(units=embedding_size, step_num=step_num, name='graph_conv')([actions_nodes, edges])

    # pick embeddings for sequence of actions with shape (NUMBER_OF_ACTIONS, embedding_size)
    embeddings_actions = Lambda(lookup_embeddings, output_shape=(NUMBER_OF_ACTIONS, embedding_size), name="embeddings_actions")([actions, graph_conv])
    
    # lstm layer
    lstm = LSTM(512, return_sequences=False, input_shape=(NUMBER_OF_ACTIONS, embedding_size), name='lstm')(embeddings_actions)

    # classification layer fully connected with dropout
    dense_1 = Dense(1024, activation='relu', name='dense_1')(lstm)
    drop_1 = Dropout(0.8, name='drop_1')(dense_1)
    dense_2 = Dense(1024, activation='relu', name='dense_2')(drop_1)
    drop_2 = Dropout(0.8, name='drop_2')(dense_2)
    output_action = Dense(NUMBER_OF_DISTINCT_ACTIONS, activation='softmax', name='output_action')(drop_2)

    return Model([actions, actions_nodes, edges], output_action)


# G2 GCN-CNN architecture (two branches and pick embeddings for the sequence after graph convolution)
def GCN_CNN_MODEL(NUMBER_OF_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS, embedding_size, step_num):
    # input layer with action sequences sliding window
    actions = Input(shape=(NUMBER_OF_ACTIONS,), name='actions', dtype=tf.int32)

    # input layer with unique actions (with word2vec embeddings as features) and adjacency matrix
    actions_nodes = Input(shape=(NUMBER_OF_DISTINCT_ACTIONS, embedding_size), name='actions_nodes')
    edges = Input(shape=(NUMBER_OF_DISTINCT_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS), name='edges')

    # graph conv layer which outputs a tensor with shape (NUMBER_OF_DISTINCT_ACTIONS, embedding_size)
    graph_conv = GraphConv(units=embedding_size, step_num=step_num, name='graph_conv')([actions_nodes, edges])

    # pick embeddings for sequence of actions with shape (NUMBER_OF_ACTIONS, embedding_size)
    embeddings_actions = Lambda(lookup_embeddings, output_shape=(NUMBER_OF_ACTIONS, embedding_size), name="embeddings_actions")([actions, graph_conv])
    reshape = Reshape((NUMBER_OF_ACTIONS, embedding_size, 1), name='reshape')(embeddings_actions)

    # convolutional layer
    ngram_2 = Convolution2D(200, (2, embedding_size), padding='valid', activation='relu', name='conv_2')(reshape)
    maxpool_2 = MaxPooling2D(pool_size=(NUMBER_OF_ACTIONS-2+1,1), name='pooling_2')(ngram_2)
    ngram_3 = Convolution2D(200, (3, embedding_size), padding='valid', activation='relu', name='conv_3')(reshape)
    maxpool_3 = MaxPooling2D(pool_size=(NUMBER_OF_ACTIONS-3+1,1), name='pooling_3')(ngram_3)
    ngram_4 = Convolution2D(200, (4, embedding_size), padding='valid', activation='relu', name='conv_4')(reshape)
    maxpool_4 = MaxPooling2D(pool_size=(NUMBER_OF_ACTIONS-4+1,1), name='pooling_4')(ngram_4)
    ngram_5 = Convolution2D(200, (5, embedding_size), padding='valid', activation='relu', name='conv_5')(reshape)
    maxpool_5 = MaxPooling2D(pool_size=(NUMBER_OF_ACTIONS-5+1,1), name='pooling_5')(ngram_5)
    merged = Concatenate(axis=2)([maxpool_2, maxpool_3, maxpool_4, maxpool_5])
    flatten = Flatten(name='flatten')(merged)

    # classification layer fully connected with dropout
    dense_1 = Dense(512, activation='relu', name='dense_1')(flatten)
    drop_1 = Dropout(0.8, name='drop_1')(dense_1)
    dense_2 = Dense(512, activation='relu', name='dense_2')(drop_1)
    drop_2 = Dropout(0.8, name='drop_2')(dense_2)
    output_action = Dense(NUMBER_OF_DISTINCT_ACTIONS, activation='softmax', name='output_action')(drop_2)

    return Model([actions, actions_nodes, edges], output_action)

# GCN architecture TEST
def GCN_MODEL_ONLY_EMB(NUMBER_OF_DISTINCT_ACTIONS, embedding_size):
    # input layer with distinc actions and adyacency matrix
    actions_nodes = Input(shape=(NUMBER_OF_DISTINCT_ACTIONS, embedding_size), name='actions')
    edges = Input(shape=(NUMBER_OF_DISTINCT_ACTIONS, NUMBER_OF_DISTINCT_ACTIONS), name='edges')

    # graph conv layer
    graph_conv = GraphConv(units=embedding_size, step_num=1, name='graph_conv')([actions_nodes, edges])

    return Model([actions_nodes, edges], graph_conv)