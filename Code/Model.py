# %%
import tensorflow as tf
import numpy as np
import copy
import pandas as pd

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, batch_sz, **kwargs):
        super().__init__(self, **kwargs)

        self.batch_sz = batch_sz
        self.vocab_size = vocab_size
        self.embedding_dim = 128
        self.units = 64

        self.embedding_layer = tf.keras.layers.Embedding(input_dim = self.vocab_size, output_dim = self.embedding_dim)
        self.gru_layer = tf.keras.layers.GRU(units = self.units, return_sequences = True, return_state = True)
        
    def call(self, input_seq, enc_hidden):
        x = self.embedding_layer(input_seq)
        output, state = self.gru_layer(x)

        return output, state

    def initialize_hidden_state(self):
        return tf.zeros([self.batch_sz, self.units])

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(self, **kwargs)
        
        self.vocab_size = vocab_size
        self.embedding_dim = 128
        self.units = 64

        self.embedding_layer = tf.keras.layers.Embedding(input_dim = self.vocab_size, output_dim = self.embedding_dim)
        self.gru_layer = tf.keras.layers.GRU(units = self.units, return_sequences = True)
        self.softmax_layer = tf.keras.layers.Dense(units = self.vocab_size, activation = 'softmax')

    def call(self, target, dec_hidden):
        target = tf.reshape(target, shape = (-1, 1))
        x = self.embedding_layer(target)
        dec_hidden = self.gru_layer(x, initial_state = dec_hidden)  # return hidden_state
        outputs = self.softmax_layer(dec_hidden)    # predict probability

        return tf.squeeze(outputs), tf.squeeze(dec_hidden)

def train_seq2seq(x, enc_hidden):
    ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction = 'none')
    optimizer = tf.keras.optimizers.Adam(1e-3)


    inputs = x[:, :x.shape[1] - 1]
    targets = x[:, 1:x.shape[1]]
    seqs = tf.reshape(inputs[:, 0], shape = (-1, 1))

    with tf.GradientTape() as tape:

        enc_output, enc_hidden = encoder(inputs, enc_hidden)
        dec_hidden = copy.deepcopy(enc_hidden)
        total_loss = 0

        trainable_variables = encoder.trainable_variables + decoder.trainable_variables

        total_score = 0
        pre_score = np.zeros([inputs.shape[0]])
        for t in range(inputs.shape[1]):
            # print('t :', t)
            preds, dec_hidden = decoder(inputs[:, t], dec_hidden)

            # 매 t마다 손실 계산
            tars = tf.reshape(targets[:, t], shape = (-1, 1))
            loss = ce_loss(tars, preds)
            total_loss += loss  # 누적 손실

            # 타겟 상태 업데이트
            seqs = np.append(seqs, tars, axis = 1)

            # # (Option 1) 매 t마다 보상 계산
            # nutrition_state = np.apply_along_axis(get_score_vector, arr = seqs, axis = 1, nutrient_data = nutrient_data)
            # score = np.apply_along_axis(get_reward_ver2, arr = nutrition_state, axis = 1, done = 0)
            # total_score += score[:, 0] - pre_score  # 누적 보상
            # pre_score = copy.deepcopy(score[:, 0]) # pre_score 재할당

        # (Option 2) 전체 sequence에 대해 보상 계산        
        nutrition_state = np.apply_along_axis(get_score_vector, arr = seqs, axis = 1, nutrient_data = nutrient_data)
        reward = np.apply_along_axis(get_reward_ver2, arr = nutrition_state, axis = 1, done = 0)
        reward = reward[:, 0]

        # 본 배치의 평균 손실
        batch_loss = (tf.reduce_mean(total_loss).numpy() / int(targets.shape[1]))

        # 정책 그라디언트 형태로 그라디언트 계산 (누적 로스 * 누적 보상)
        # grads = tape.gradient(total_loss * total_score, trainable_variables)
        grads = tape.gradient(total_loss * reward, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))

    return batch_loss


# def train_opac(self):

# %%
