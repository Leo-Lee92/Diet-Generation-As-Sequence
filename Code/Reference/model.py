# %%
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# # 모듈로 돌릴 때 필요한 것 패키지
# from preprocessing import food_dict

class RNN(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

        # weight 초기화 시드 고정
        self.seed_nb = 1234
        tf.random.set_seed(self.seed_nb)
        os.environ['PYTHONHASHSEED'] = str(self.seed_nb)
        np.random.seed(self.seed_nb)
        random.seed(self.seed_nb)

        self.action_size = len(food_dict)
        self.embedding_dim = 200
        self.encoding_units = 512 # hidden_layer 인코딩하는 뉴런 개수

        self.embedding_layer = tf.keras.layers.Embedding(input_dim = self.action_size, output_dim = self.embedding_dim, embeddings_initializer = tf.keras.initializers.glorot_uniform(seed = self.seed_nb))
        self.gru_layer = tf.keras.layers.GRU(units = self.encoding_units, return_sequences= True, return_state= True, kernel_initializer = tf.keras.initializers.glorot_uniform(seed = self.seed_nb))
        self.rnn_softmax_layer = tf.keras.layers.Dense(units = self.action_size, activation = 'softmax', kernel_initializer = tf.keras.initializers.glorot_uniform(seed = self.seed_nb))

    def initialize_hidden_state(self, BATCH_SIZE):
        return tf.zeros([BATCH_SIZE, self.encoding_units])

    def call(self, inputs, h_state):
        
        X = self.embedding_layer(inputs)
        rnn_outputs, rnn_state = self.gru_layer(X, initial_state = h_state)
        rnn_outputs = tf.reshape(rnn_outputs, shape = (-1, rnn_outputs.shape[2]))   # X_outputs은 3D 텐서이며, [batch, token_length, hidden_size] 크기를 가진다. 
                                                                                    # token_length는 1이므로 tf.reshape로 2번 차원만 고정하여 1번 차원을 없애줌        
        rnn_outputs = self.rnn_softmax_layer(rnn_outputs)                           # 마지막 layer (policy vector). 현재 인풋 토큰의 다음 토큰으로 어떤 토큰이 나올지 예측하는 층.

        return tf.squeeze(rnn_outputs), rnn_state

class RLActor(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

        # weight 초기화 시드 고정
        self.seed_nb = 1234
        tf.random.set_seed(self.seed_nb)
        os.environ['PYTHONHASHSEED'] = str(self.seed_nb)
        np.random.seed(self.seed_nb)
        random.seed(self.seed_nb)

        self.embedding_dim = 200
        self.encoding_dim = 512
        self.action_size = len(food_dict)
        self.embedding_layer = tf.keras.layers.Embedding(input_dim = self.action_size, output_dim = self.embedding_dim, embeddings_initializer = tf.keras.initializers.glorot_uniform(seed = self.seed_nb))
        self.dense_layer = tf.keras.layers.Dense(units = self.encoding_dim, kernel_initializer = tf.keras.initializers.glorot_uniform(seed = self.seed_nb))
        self.softmax_layer = tf.keras.layers.Dense(units = self.action_size, activation = 'softmax', kernel_initializer = tf.keras.initializers.glorot_uniform(seed = self.seed_nb))

    def initialize_hidden_state(self, BATCH_SIZE):
        return tf.zeros([BATCH_SIZE, self.encoding_dim])

    def call(self, inputs):

        '''
        Simple MLP
        '''
        X = self.embedding_layer(inputs)
        X = tf.squeeze(X)
        X = self.dense_layer(X)
        X = self.softmax_layer(X)
        
        return tf.squeeze(X)

class RLCritic(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

        # weight 초기화 시드 고정
        self.seed_nb = 1234
        tf.random.set_seed(self.seed_nb)
        os.environ['PYTHONHASHSEED'] = str(self.seed_nb)
        np.random.seed(self.seed_nb)
        random.seed(self.seed_nb)

        self.action_size = len(food_dict)
        self.embedding_layer = tf.keras.layers.Embedding(input_dim = self.action_size, output_dim = 200, embeddings_initializer = tf.keras.initializers.glorot_uniform(seed = self.seed_nb))
        self.dense_layer = tf.keras.layers.Dense(units = 1, kernel_initializer = tf.keras.initializers.glorot_uniform(seed = self.seed_nb))

    def initialize_hidden_state(self, BATCH_SIZE):
        return tf.zeros([BATCH_SIZE, 256])

    def call(self, inputs):
        X = self.embedding_layer(inputs)
        X = self.dense_layer(X)

        return tf.squeeze(X)

class Q_Network(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)
        # weight 초기화 시드 고정
        self.seed_nb = 1234
        tf.random.set_seed(self.seed_nb)
        os.environ['PYTHONHASHSEED'] = str(self.seed_nb)
        np.random.seed(self.seed_nb)
        random.seed(self.seed_nb)

        self.action_size = len(food_dict)
        self.embedding_dim = 200
        self.encoding_units = 512 # hidden_layer 인코딩하는 뉴런 개수

        self.embedding_layer = tf.keras.layers.Embedding(input_dim = self.action_size, output_dim = self.embedding_dim, embeddings_initializer = tf.keras.initializers.glorot_uniform(seed = self.seed_nb))
        self.gru_layer = tf.keras.layers.GRU(units = self.encoding_units, return_sequences= True, return_state= True, kernel_initializer = tf.keras.initializers.glorot_uniform(seed = self.seed_nb))
        self.dense_layer = tf.keras.layers.Dense(units = self.action_size, kernel_initializer = tf.keras.initializers.glorot_uniform(seed = self.seed_nb))

    def initialize_hidden_state(self, BATCH_SIZE):
        return tf.zeros([BATCH_SIZE, self.encoding_units])

    def call(self, inputs, h_state):
        X = self.embedding_layer(inputs)
        rnn_outputs, rnn_state = self.gru_layer(X, initial_state = h_state)
        X = tf.squeeze(rnn_outputs)
        X = self.dense_layer(X)

        return tf.squeeze(X), rnn_state
        
# %%
