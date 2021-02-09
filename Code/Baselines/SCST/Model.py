# %%
import tensorflow as tf
import numpy as np
import copy
import pandas as pd
import sys
sys.path.append("/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code")
from util import *
from Preprocessing import *


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

class Attention(tf.keras.Model):
    def __init__(self, units, **kwargs):    # units은 decoder로부터 상속됨.
        super().__init__(self, **kwargs)
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
    def __init__(self, vocab_size, **kwargs):
        super().__init__(self, **kwargs)
        
        self.vocab_size = vocab_size
        self.embedding_dim = 128
        self.units = 64

        self.embedding_layer = tf.keras.layers.Embedding(input_dim = self.vocab_size, output_dim = self.embedding_dim)
        self.gru_layer = tf.keras.layers.GRU(units = self.units, return_sequences = True, return_state = True)
        self.softmax_layer = tf.keras.layers.Dense(units = self.vocab_size, activation = 'softmax')

        self.Attention = Attention(self.units)

    def call(self, target, dec_hidden, enc_output):
        # (1) 타겟 (토큰) 데이터에 1번 차원 추가
        target = tf.reshape(target, shape = (-1, 1))

        # (2) 타겟 (토큰) 데이터의 임베딩 벡터 반환
        x = self.embedding_layer(target)

        # (3) 현재 dec_hidden에 대해 Attention 적용하여 context_vector 반환
        context_vector, attention_weights = self.Attention(dec_hidden, enc_output)

        # (4) 반환된 context_vector 차원 추가 및 임베딩 벡터와 context_vector의 concat
        ## embedding_vector인 x는 3축 텐서인 반면 context_vector는 2축 텐서이므로
        ## context_vector의 축을 하나 늘려준다.
        x = tf.concat([tf.expand_dims(context_vector, axis = 1), x], axis = -1)

        # (5) concat된 벡터를 통해 dec_hidden 반환
        _, dec_hidden = self.gru_layer(x)  # return hidden_state
        # dec_hidden = self.gru_layer(x, initial_state = dec_hidden)    # return hidden_state
        outputs = self.softmax_layer(dec_hidden)                        # predict probability

        # return tf.squeeze(outputs), tf.squeeze(dec_hidden, axis = 0)
        return tf.squeeze(outputs), tf.squeeze(dec_hidden), attention_weights

def train_SEQ2SEQ_for_SCST(x, encoder, decoder, epoch, lr, ss_prob):
    # 토큰 시퀀스 배치 x와 초기화 된 encoder를 받는다.
    # 초기화 된 decoer를 받는다

    # 손실함수, 옵티마이저 초기화
    ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction = 'none')
    xe_optimizer = tf.keras.optimizers.Adam(lr)

    # 인코더의 히든 스테이트 초기화
    enc_hidden = encoder.initialize_hidden_state()

    # 토큰시퀀스 데이터 구축
    inputs = x[:, :x.shape[1] - 1]
    targets = x[:, 1:x.shape[1]]

    # SL 머신이 예측하는 시퀀스 초기화
    xe_pred_seqs = tf.reshape(inputs[:, 0], shape = (-1, 1))

    # SL은 Seq2Seq 파라미터를 업데이트
    xe_trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    # 각종 파라미터 초기화
    xe_loss = 0                           # SL의 누적손실 초기화
    
    '''
    Supervised Learning (SL) with XE-loss 파트
    '''
    with tf.GradientTape(watch_accessed_variables = False) as xe_tape:
        xe_tape.watch(xe_trainable_variables)

        # 인코더의 컨텍스트 벡터 뽑아주기
        enc_output, enc_hidden = encoder(inputs, enc_hidden)

        # 히든 스테이트 정의
        dec_hidden = copy.deepcopy(enc_hidden)

        for t in range(inputs.shape[1]):
            
            '''
            SL 학습
            '''
            # SL 학습
            ## (SL-1) 예측
            if np.random.sample(1) > ss_prob:
                preds, dec_hidden, attention_weights = decoder(inputs[:, t], dec_hidden, enc_output)                
            else:
                preds, dec_hidden, attention_weights = decoder(xe_pred_seqs[:, t], dec_hidden, enc_output)

            results = np.apply_along_axis(get_action, axis = 1, arr = preds, option = 'prob')
            predicted_token = tf.reshape(results[:, 0], shape = (-1, 1))        # SL 머신이 예측한 토큰

            ## (SL-2) 손실 계산 : 매 t마다 손실 계산
            tars = tf.reshape(targets[:, t], shape = (-1, 1))
            loss = ce_loss(tars, preds)
            xe_loss += loss  # 누적 손실

            ## SL이 예측한 시퀀스 누적
            xe_pred_seqs = np.append(xe_pred_seqs, predicted_token, axis = 1)

        # (SL-3) SL 그라디언트 계산 및 업데이트
        xe_grads = xe_tape.gradient(xe_loss, xe_trainable_variables)
        xe_optimizer.apply_gradients(zip(xe_grads, xe_trainable_variables))

        # 본 배치의 평균 손실
        batch_loss = (tf.reduce_mean(xe_loss).numpy() / int(targets.shape[1]))

    return batch_loss, xe_pred_seqs

def train_SCST(x, encoder, decoder, curriculum_range, lr, reward_shaping = False):

    '''
    t: current step
    inputs: inputs to be fed into RL agent
    targets: targets to be fed into RL agent in case we use off-policy control
    '''
    # 손실함수, 옵티마이저 초기화
    ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction = 'none')

    if curriculum_range == None:
        SCST_optimizer = tf.keras.optimizers.Adam(lr)
    else:
        SCST_optimizer = tf.keras.optimizers.Adam(lr)


    # 두 인코더의 히든 스테이트 초기화
    enc_hidden = encoder.initialize_hidden_state()

    # 토큰시퀀스 데이터 구축
    inputs = x[:, :x.shape[1] - 1]
    targets = x[:, 1:x.shape[1]]

    # RL 에이전트가 예측하는 시퀀스 초기화
    pred_seqs = tf.reshape(inputs[:, 0], shape = (-1, 1))       # 예측한 시퀀스
    train_seqs = tf.reshape(inputs[:, 0], shape = (-1, 1))      # 학즙한 시퀀스 (타겟 + 예측)
    greedy_seqs = tf.reshape(inputs[:, 0], shape = (-1, 1))      # 학즙한 시퀀스 (타겟 + 예측)

    # 각종 파라미터 초기화
    SCST_loss = 0                                   # SL의 누적손실 초기화
    reward_over_timestep = 0                        # RL의 배치별 보상 초기화

    # 업데이트 파라미터 지정
    SCST_trainable_variables = decoder.trainable_variables

    # 손실 초기화
    full_SCST_loss = 0

    with tf.GradientTape(watch_accessed_variables = False) as SCST_tape:
        SCST_tape.watch(SCST_trainable_variables)
    
        # 두 인코더의 컨텍스트 벡터 각각 뽑아주기
        enc_output, enc_hidden = encoder(inputs, enc_hidden)

        # 인코더의 마지막 히든 스테이트를 디코더의 초기 히든 스테이트로 정의
        dec_hidden = copy.deepcopy(enc_hidden)

        for t in range(inputs.shape[1]):            # t: 현재 step

            # (1) 만약 curriculum learning이 전 구간에서 발생한다면 -> XE_gradient로 Learning 하라
            if curriculum_range == "All":
                # 예측
                preds, dec_hidden, _ = decoder(inputs[:, t], dec_hidden, enc_output)    
                results = np.apply_along_axis(get_action, axis = 1, arr = preds, option = 'prob')
                actions = tf.reshape(results[:, 0], shape = (-1, 1))                                    

                # 시퀀스 누적
                pred_seqs = np.append(pred_seqs, actions, axis = 1) 
                train_seqs = np.append(train_seqs, tf.reshape(targets[:, t], shape = (-1, 1)), axis = 1)        # non_curriculum_learning 중일 땐 학습중인 시퀀스가 실제 데이터와 같음
                # XENT 손실 계산
                SCST_loss = ce_loss(targets[:, t], preds)

            # (2) 만약 curriculumn learning 하지 않는다면 -> 바로 REINFORCE Learning 하라
            ## 주의. 정상적인 sequence가 나오려면 반드시 encoder-decoder가 사전학습 되어 있어야 함.
            elif curriculum_range == None:
                # RL 학습
                # 예측
                preds, dec_hidden, _ = decoder(train_seqs[:, t], dec_hidden, enc_output)                             # 자기 자신 (pred_seqs)을 인풋으로 받음
                results = np.apply_along_axis(get_action, axis = 1, arr = preds, option = 'prob')
                greedy_results = np.apply_along_axis(get_action, axis = 1, arr = preds, option = 'max')

                actions = tf.reshape(results[:, 0], shape = (-1, 1))                                        # 에이전트가 선택한 행위
                greedy_actions = tf.reshape(greedy_results[:, 0], shape = (-1, 1))                          # 에이전트가 선택한 행위

                probs = tf.reshape(results[:, 1], shape = (-1, 1))                                          # 에이전트가 선택한 행위의 확률

                # 시퀀스 누적
                pred_seqs = np.append(pred_seqs, actions, axis = 1)
                train_seqs = np.append(train_seqs, actions, axis = 1)                                       # curriculum_learning 중일 때 curriculum_range 이후엔 예측 데이터 (actions)를 입력 
                greedy_seqs = np.append(greedy_seqs, greedy_actions, axis = 1)                              # curriculum_learning 중일 때 curriculum_range 이후엔 예측 데이터 (actions)를 입력 

                # RL 손실 계산
                SCST_loss = ce_loss(actions, preds)       # REINFORCE 손실 계산                                   

            # (3) 만약 curriculumn learning 시작했는데
            else:
                # 현재 step이 curriculum_range 안에 있을 경우 (현 step은 curriculum learning 안해도 된다면) -> XE_gradient로 Learning 하라
                if t < curriculum_range:    
                    # SL 학습
                    # 예측
                    preds, dec_hidden, _ = decoder(inputs[:, t], dec_hidden, enc_output)    
                    results = np.apply_along_axis(get_action, axis = 1, arr = preds, option = 'prob')
                    actions = tf.reshape(results[:, 0], shape = (-1, 1))                                  

                    # 시퀀스 누적
                    pred_seqs = np.append(pred_seqs, actions, axis = 1)
                    train_seqs = np.append(train_seqs, tf.reshape(targets[:, t], shape = (-1, 1)), axis = 1)          # curriculum_learning 중일 때 curriculum_range까진 실제 데이터 (targets)를 입력
                    
                    # XENT 손실 계산
                    SCST_loss = ce_loss(targets[:, t], preds)                                             
                
                # 현재 step이 curriculum_range 밖에 있을 경우 (현 step에선 curriculum learning 해야 한다면) -> policy_gradient로 Learning 하라
                else:   
                    # RL 학습
                    # 예측
                    preds, dec_hidden, _ = decoder(train_seqs[:, t], dec_hidden, enc_output)                    # 자기 자신 (pred_seqs)을 인풋으로 받음
                    results = np.apply_along_axis(get_action, axis = 1, arr = preds, option = 'prob')
                    greedy_results = np.apply_along_axis(get_action, axis = 1, arr = preds, option = 'max')

                    actions = tf.reshape(results[:, 0], shape = (-1, 1))                                        # 에이전트가 선택한 행위
                    greedy_actions = tf.reshape(greedy_results[:, 0], shape = (-1, 1))                          # 에이전트가 선택한 행위

                    probs = tf.reshape(results[:, 1], shape = (-1, 1))                                          # 에이전트가 선택한 행위의 확률

                    # 시퀀스 누적
                    pred_seqs = np.append(pred_seqs, actions, axis = 1)
                    train_seqs = np.append(train_seqs, actions, axis = 1)                                       # curriculum_learning 중일 때 curriculum_range 이후엔 예측 데이터 (actions)를 입력 
                    greedy_seqs = np.append(greedy_seqs, greedy_actions, axis = 1)                              # curriculum_learning 중일 때 curriculum_range 이후엔 예측 데이터 (actions)를 입력 

                    # RL 손실 계산
                    SCST_loss = ce_loss(actions, preds)       # REINFORCE 손실 계산

            full_SCST_loss += SCST_loss
            
        #  손실에 보상 반영여부 결정
        if curriculum_range == None:    # 만약 curriculum learning을 시작 이후라면 보상을 반영한 손실 (XE+R)

            # reward_shaping 여부 결정
            if reward_shaping == "False":
                # nutrition_state = np.apply_along_axis(get_score_vector, arr = train_seqs, axis = 1, nutrient_data = nutrient_data)
                nutrition_state = get_score_matrix(train_seqs, nutrient_data, food_dict)
                score = np.apply_along_axis(get_reward_ver2, arr = nutrition_state, axis = 1, done = 0)

                # nutrition_state = np.apply_along_axis(get_score_vector, arr = greedy_seqs, axis = 1, nutrient_data = nutrient_data)
                nutrition_state = get_score_matrix(greedy_seqs, nutrient_data, food_dict)
                infer_score = np.apply_along_axis(get_reward_ver2, arr = nutrition_state, axis = 1, done = 0)
            else:
                # (1-1) train_seqs에 대해 영양 점수 계산
                # nutrition_state = np.apply_along_axis(get_score_vector, arr = train_seqs, axis = 1, nutrient_data = nutrient_data)
                nutrition_state = get_score_matrix(train_seqs, nutrient_data, food_dict)
                score = np.apply_along_axis(get_reward_ver2, arr = nutrition_state, axis = 1, done = 0)[:, 0]

                # (1-2) train_seqs에 대해 구성 점수 반영
                score *= np.apply_along_axis(meal_hit_score, arr = train_seqs, axis = 1, category_data = category_data)[:, 0] * np.apply_along_axis(dish_hit_score, arr = train_seqs, axis = 1, category_data = category_data)[:, 0]
                # score *= np.apply_along_axis(get_composition_score, arr = train_seqs, axis = 1, category_data = category_data)

                # (2-1) greedy_seqs에 대해 영양 점수 계산
                # nutrition_state = np.apply_along_axis(get_score_vector, arr = greedy_seqs, axis = 1, nutrient_data = nutrient_data)
                nutrition_state = get_score_matrix(greedy_seqs, nutrient_data, food_dict)
                infer_score = np.apply_along_axis(get_reward_ver2, arr = nutrition_state, axis = 1, done = 0)[:, 0]
    
                # (2-2) greedy_seqs에 대해 구성 점수 반영
                infer_score *= np.apply_along_axis(meal_hit_score, arr = greedy_seqs, axis = 1, category_data = category_data)[:, 0] * np.apply_along_axis(dish_hit_score, arr = greedy_seqs, axis = 1, category_data = category_data)[:, 0]
                # infer_score *= np.apply_along_axis(get_composition_score, arr = greedy_seqs, axis = 1, category_data = category_data)

            reward = score - infer_score         # 보상            
            final_loss = full_SCST_loss * reward

        else:   # 만약 curriculum learning을 시작 전이라면 순전히 XENT만 고려한 손실 (XENT)
            final_loss = full_SCST_loss     

    true_reward = score.reshape(-1, 1)

    ## (RL-5) RL 그라디언트 계산 및 업데이트
    SCST_grads = SCST_tape.gradient(final_loss, SCST_trainable_variables)
    SCST_optimizer.apply_gradients(zip(SCST_grads, SCST_trainable_variables))

    # 본 배치의 평균 손실
    batch_loss = (tf.reduce_mean(full_SCST_loss).numpy() / int(inputs.shape[1]))

    '''
    batch_loss : RL 에이전트의 배치별 평균 손실
    pred_seqs : RL 에이전트가 예측한 trajectory
    greedy_seqs : RL 에이전트가 greedy하게 추정한 trajectory
    '''
    return batch_loss, pred_seqs, greedy_seqs, true_reward

# %%
