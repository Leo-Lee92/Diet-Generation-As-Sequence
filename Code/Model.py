
# %%
from util import get_action, get_score_matrix, get_reward_ver2, get_score_vector, make_synthetic_target

import tensorflow as tf
import numpy as np
import copy
import pandas as pd

class Encoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self)

        self.batch_sz = kwargs['batch_size']
        self.vocab_size = kwargs['num_tokens']
        self.embedding_dim = kwargs['embed_dim']
        self.units = kwargs['fc_dim']
        self.layer_type = kwargs['fully-connected_layer']
        
        # Define embedding layer.
        self.embedding_layer = tf.keras.layers.Embedding(input_dim = self.vocab_size, output_dim = self.embedding_dim)

        # Define fully-connected layer.
        if self.layer_type == 'GRU':
            self.fc_layer = tf.keras.layers.GRU(units = self.units, return_sequences = True, return_state = True)
        elif self.layer_type == 'LSTM':
            self.fc_layer = tf.keras.layers.LSTM(units = self.units, return_sequences = True, return_state = True)
        
    def call(self, input_seq, enc_hidden):
        x = self.embedding_layer(input_seq)

        if self.layer_type == 'GRU':
            output, state = self.fc_layer(x)
        elif self.layer_type == 'LSTM':
            output, state, _ = self.fc_layer(x)

        return output, state

    def initialize_hidden_state(self):
        return tf.zeros([self.batch_sz, self.units])

class Attention(tf.keras.Model):
    def __init__(self, units, **kwargs):    # the argment 'units' inherits from decoder.
        super().__init__(self)
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_hidden, all_enc_hiddens):
        # dec_hidden : query
        # all_enc_hiddens : enc_output or values 

        # (1) dec_hidden에 1번 차원 추가
        ## 참고로, 0번 차원 = batch, 1번 차원 = time (step), 2번 차원 = feature (token) 임.
        ## 1번차원을 추가하는 이유는, 밑에서 bahdanau_additive를 계산할 때 query와 values를 더하게 되는데
        ## 이 때, query는 현 step t에 대한 값인데 반해 values는 모든 t에 대한 텐서라서 시간축이 존재하기 때문에
        ## bahdanau_additive를 계산하기 위해선 차원을 맞춰줘야 함 (자동으로 broadcasting 시켜주기 위함인듯?).
        query_with_time_axis = tf.expand_dims(dec_hidden, 1)

        # (2) dec_hidden (decoder의 state)와 all_enc_hiddens (enc_output; encoder의 all states)를 더하기
        bahdanau_additive = self.W1(query_with_time_axis) + self.W2(all_enc_hiddens)
 
        # (3) bahdanau_additive에 tanh 함수를 적용하여 attention_score로서 활성화
        ## tanh함수로 활성화 함으로써 bahdanau_additive의 범위는 [-1, 1]로 정규화 된다.
        ## sigmoid와 비교했을때, tanh는 출력범위가 더 넓고 경사면이 학습과 수렴 속도가 더 빨라짐.
        attention_score = self.V(tf.nn.tanh(bahdanau_additive))

        # (4) 1번 차원 (시간축)을 기준으로 attention_score를 softmax 함수를 적용하여 probability로서 활성화
        ## 즉, attention_weigths의 의미는 디코더의 현재 dec_hidden가
        ## 시간축을 따라 펼쳐진 인코더의 all_enc_hiddens들에 대해서 갖는 확률적 관련성을 의미함.
        attention_weights = tf.nn.softmax(attention_score, axis = 1)

        # (5) 가중합을 통한 context_vector 구하기
        ## attention_weights는 dec_hidden과 all_enc_hiddens 간의 관련성이다.
        ## 시간축을 따라 attention_weights * all_enc_hiddens 한다는 것은 dec_hidden과 관련성이 높은 enc_hidden(t)에 높은 가중치를 부여하겠다는 뜻.
        ## 즉 context_vector는 dec_hidden과 관련이 enc_hidden에 '주의 (어텐션)'한 정보벡터가 됨.
        context_vector = tf.reduce_sum(attention_weights * all_enc_hiddens, axis = 1)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self)
        
        self.vocab_size = kwargs['num_tokens']
        self.embedding_dim = kwargs['embed_dim']
        self.units = kwargs['fc_dim']
        self.use_attention = kwargs['attention']
        self.layer_type = kwargs['fully-connected_layer']

        # Define embedding layer.
        self.embedding_layer = tf.keras.layers.Embedding(input_dim = self.vocab_size, output_dim = self.embedding_dim)

        # Define fully-connected layer.
        if self.layer_type == 'GRU':
            self.fc_layer = tf.keras.layers.GRU(units = self.units, return_sequences = True, return_state = True)
        elif self.layer_type == 'LSTM':
            self.fc_layer = tf.keras.layers.LSTM(units = self.units, return_sequences = True, return_state = True)

        # Define softmax layer
        self.softmax_layer = tf.keras.layers.Dense(units = self.vocab_size, activation = 'softmax')

        # Define attention layer
        if self.use_attention == True:
            self.Attention = Attention(self.units, **kwargs)            

    def call(self, target, dec_hidden, enc_output):
        # (1) 타겟 (토큰) 데이터에 1번 차원 추가
        target = tf.reshape(target, shape = (-1, 1))

        # (2) 타겟 (토큰) 데이터의 임베딩 벡터 반환
        x = self.embedding_layer(target)

        # if we use attention mechanism
        if self.use_attention == True:

            # (step 1) Run attention in terms of the current dec_hidden and return context vector.
            context_vector, attention_weights = self.Attention(dec_hidden, enc_output)
    
            # (step 2) Concat context vector with embedding vector x.
            x = tf.concat([tf.expand_dims(context_vector, axis = 1), x], axis = -1)

            # (step 3) Return new dec_hidden by passing the above concatenated vector into fully-connected layer.
            if self.layer_type == 'GRU':
                _, dec_hidden = self.fc_layer(x)  # return hidden_state
            elif self.layer_type == 'LSTM':
                _, dec_hidden, _ = self.fc_layer(x)  # return hidden_state

        # if we don't use attention mechanism
        else:
            # (step 1) Return new dec_hidden by passing the dec_hidden and embedding vector together.
            ## Note that we have to explicitly set the initial state every t-th step if you want to execute RNN-based network using for loop just as what it is in executing Attention-based network.
            if self.layer_type == 'GRU':
                _, dec_hidden = self.fc_layer(x, initial_state = dec_hidden)  # return hidden_state
            elif self.layer_type == 'LSTM':
                _, dec_hidden, _ = self.fc_layer(x, initial_state = [dec_hidden, dec_hidden])  # return hidden_state

        # Pass dec_hidden into softmax layer and obtain outputs (the probability distribution).
        outputs = self.softmax_layer(dec_hidden)                        

        # Return the outputs, dec_hidden, and (if use_attention == True) weights of attention.
        if self.use_attention == True:
            return tf.squeeze(outputs), tf.squeeze(dec_hidden), attention_weights
        else:
            return tf.squeeze(outputs), tf.squeeze(dec_hidden), None

class Sequence_Generator(tf.keras.Model):
    def __init__(self, food_dict, nutrient_data, incidence_data, **kwargs):
        super().__init__(self)

        # Define global variables
        self.food_dict = food_dict
        self.nutrient_data = nutrient_data
        # self.incidence_mat = kwargs['incidence_matrix'].numpy()
        # self.incidence_mat[self.incidence_mat > 0] = 1
        self.incidence_mat = incidence_data.numpy()
        self.incidence_mat[self.incidence_mat > 0] = 1

        # Inherits the parameters from user-defined parameter list.
        self.policy_learning_type = kwargs['learning']
        self.policy_type = kwargs['policy']
        self.add_breakfast = kwargs['add_breakfast']
        self.lr = kwargs['lr']


        self.use_beta_score = kwargs['use_beta']
        self.synthetic_target_buffer = kwargs['use_buffer']

        # Define loss function and optimizer
        self.ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction = 'none')
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        # Define encoder and decoder.
        ## Generate inital states for input, hidden, and concat state in Encoder and Decoder.
        ## --- (1) Define Encoder that embeds food sequences
        self.encoder = Encoder(**kwargs)
        self.decoder = Decoder(**kwargs)

    def train(self, x, x_update):

        # Initialize and define the hidden states of encoder
        enc_hidden = self.encoder.initialize_hidden_state()

        # Define input and target sequences to be used in training model.
        inputs = x[:, :x.shape[1] - 1]
        targets = x_update[:, 1:x_update.shape[1]]
        tar_len = targets.shape[1]

        # Initialize and define both real and synthetic diet sequences.
        real_seqs = tf.reshape(inputs[:, 0], shape = (-1, 1))
        pred_seqs = tf.reshape(inputs[:, 0], shape = (-1, 1))

        # Define trainable_variables including the parameters both from encoder and decoder.
        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        # Initialize parameters
        total_loss = 0
        beta_score = 0

        with tf.GradientTape() as tape:

            # Get output (output vector at each state) and hidden state (aka context vector) of encoder.
            enc_output, enc_hidden = self.encoder(inputs, enc_hidden)

            # Define intial hidden state of decoder copying context vector of encoder.
            dec_hidden = copy.deepcopy(enc_hidden)
            
            for t in range(inputs.shape[1]):

                # Define real targets based on real data.
                tars = tf.reshape(targets[:, t], shape = (-1, 1))

                '''
                Train Sequence Generator
                '''
                # In case we use off-policy learning.
                ## Predict next action according to 'policy_type', but sample next action (i.e., next token) from 'inputs'.
                if self.policy_learning_type == 'off-policy':

                    ## (step 1) predict next token.
                    preds, dec_hidden, _ = self.decoder(inputs[:, t], dec_hidden, enc_output)
                    results = np.apply_along_axis(get_action, axis = 1, arr = np.array(preds), option = self.policy_type)
                    predicted_token = tf.reshape(results[:, 0], shape = (-1, 1))

                    ## (step 2) get target token under off-policy setting.
                    off_tars = tf.reshape(targets[:, t], shape = (-1, 1))

                    ## (step 3) compute step loss (e.g., categorical cross entropy) between predicted token and target token at t-th step.
                    loss = self.ce_loss(off_tars, preds)

                # In case we use on-policy learning.
                ## Predict next action according to 'policy_type', and sample next action (i.e., next token) from 'policy_type'.
                elif self.policy_learning_type == 'on-policy':  

                    ## (step 1) predict next token.                
                    if t == 0:
                        preds, dec_hidden, _ = self.decoder(inputs[:, t], dec_hidden, enc_output)

                    else:
                        on_tars = tf.squeeze(on_tars)
                        preds, dec_hidden, _ = self.decoder(on_tars, dec_hidden, enc_output)
                        
                    results = np.apply_along_axis(get_action, axis = 1, arr = np.array(preds), option = self.policy_type)
                    predicted_token = tf.reshape(results[:, 0], shape = (-1, 1))

                    ## (step 2) get target token under on-policy setting.
                    on_tars = predicted_token     
                    
                    ## (step 3) compute step loss (e.g., categorical cross entropy) between predicted token and target token at t-th step.
                    loss = self.ce_loss(on_tars, preds)

                ## (step 4) compute total loss by cumulating step losses over every t-th token.
                total_loss += loss

                ## (step 5) compute beta score
                if self.use_beta_score == True:
                    ## incidence_mat : incidence matrix of which a food represents each node and a slot represents each link.
                    indicator = self.incidence_mat[np.array(predicted_token, dtype = int), t + 1]
                    beta_score += indicator
                else:
                    # Let beta_score fix to length of target sequence as this score is normalized by the length of sequence.
                    beta_score = tf.constant(tar_len, dtype = tf.float32)    

                '''
                Collect the generated diets
                '''
                ## real_seqs : sequence of target menu tokens (i.e., real diets)
                ## pred_seqs : sequence of predicted menu tokens (i.e., generated diets)
                real_seqs = np.append(real_seqs, tars, axis = 1)
                pred_seqs = np.append(pred_seqs, predicted_token, axis = 1)

            '''
            Compute reward
            '''
            ## (step 1) compute nutrition score of real diets.
            # nutrition_real = get_score_matrix(real_seqs, self.food_dict, self.nutrient_data)
            nutrition_gen = get_score_matrix(pred_seqs, self.food_dict, self.nutrient_data)

            ## (step 2) compute reward using nutrition score.
            # reward_real = np.apply_along_axis(get_reward_ver2, arr = nutrition_real, axis = 1, done = 0, mode = self.add_breakfast)
            # reward_real = tf.cast(reward_real[:, 0].astype(float), dtype = tf.float32)
            reward_gen = np.apply_along_axis(get_reward_ver2, arr = nutrition_gen, axis = 1, done = 0, mode = self.add_breakfast)
            reward_gen = tf.cast(reward_gen[:, 0].astype(float), dtype = tf.float32)

            ## (step 3) compute final reward adding beta score.
            # final_reward = reward_real * (tf.reshape(beta_score, shape = (-1, )) / tar_len)
            final_reward = reward_gen * (tf.reshape(beta_score, shape = (-1, )) / tar_len)

            '''
            Compute loss
            '''
            ## (step 4) compute final loss by element-wise product of total loss and final reward.
            ## This step works similar to the way of MRT (Minimum Risk Training) computing loss.
            final_loss = total_loss * final_reward

            '''
            Synthesizing targets
            '''
            # if synthetic_target_buffer == True, then let's make synthetic target data.
            if self.synthetic_target_buffer == True:
                synthetic_target = make_synthetic_target(x, beta_score, pred_seqs)

            # else, then let's use origianl target data just as what it is.
            else:
                synthetic_target = copy.deepcopy(x)

        '''
        Compute and update gradients
        '''
        grads = tape.gradient(final_loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        '''
        Get batch loss
        '''
        batch_loss = (tf.reduce_mean(total_loss).numpy() / int(targets.shape[1]))

        return real_seqs, batch_loss, pred_seqs, _, _, synthetic_target, _
    
    def inference(self, input_seqs):
        '''
        input_seqs : encoder에 입력되는 input_seqs 데이터
        encoder, decoder : checkpoint에 저장되어 있는 사전에 학습된 encoder, decoder 모델 및 파라미터. initialized status로 입력받아야 함.
        '''
        # Define the length of input sequence.
        input_seq_len = input_seqs.shape[1] - 1

        # Initialize encoder and decoder.
        enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden = self.encoder(input_seqs, enc_hidden)
        dec_hidden = copy.deepcopy(enc_hidden)

        # Make empty sequence, named 'gen_seqs', of which the shape is (0, 1), and stack 'bos' as a first token of gen_seqs.
        gen_seqs = np.empty((0, 1))
        gen_seqs = np.concatenate([gen_seqs, tf.reshape(input_seqs[:, 0], shape = (-1, 1))])

        # Predict next token at each step based on the previously predicted tokens until 'eos' token is sampled.
        for j in range(input_seq_len):
            outputs, dec_hidden, attention_weigths = self.decoder(input_seqs[:, j], dec_hidden, enc_output)
            results = np.apply_along_axis(get_action, axis = 1, arr = outputs, option = 'target')
            next_token = tf.reshape(results[:, 0], shape = (-1, 1))
            gen_seqs = np.concatenate([gen_seqs, next_token], axis = 1)

        return gen_seqs

# %%
