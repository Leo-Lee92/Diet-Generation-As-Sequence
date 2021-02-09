# %%
import sys
sys.path.append("/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code")
from util import *
from Preprocessing import *

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

def train_SEQ2SEQ_TFR(x, x_update, encoder, decoder, lr, reward_shaping = False, alergy_const = False):  # MIXER, SCST 계열보단 Professor Forcing에 가깝

    # 토큰 시퀀스 배치 x와 초기화 된 encoder를 받는다.
    # 클러스터 시퀀스 배치 c와 초기화 된 c_encdoer를 받는다.
    # 초기화 된 decoer를 받는다
    '''
    incidence_mat 자연스럽게 함수안으로 넣는 방법 고안해야함
    '''
    incidence_mat = np.array(incidence_data)
    incidence_mat[incidence_mat > 0] = 1
    p_prob_mat = transition_matrix(diet_data_np, food_dict)

    # 손실함수, 옵티마이저 초기화
    ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction = 'none')
    optimizer = tf.keras.optimizers.Adam(lr)

    # 인코더의 히든 스테이트 초기화
    enc_hidden = encoder.initialize_hidden_state()

    # 토큰시퀀스 데이터 구축
    inputs = x[:, :x.shape[1] - 1]
    targets = x_update[:, 1:x_update.shape[1]]


    # 실제 시퀀스 초기화
    real_seqs = tf.reshape(inputs[:, 0], shape = (-1, 1))

    # 예측 시퀀스 초기화
    pred_seqs = tf.reshape(inputs[:, 0], shape = (-1, 1))

    # SL은 Seq2Seq 파라미터를 업데이트
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    # 각종 파라미터 초기화
    total_loss = 0                           # SL의 누적손실 초기화
    beta_score = 0
    importance_weight = 1
    p_prob_cumul = 1

    with tf.GradientTape() as tape:

        # 인코더의 컨텍스트 벡터 뽑아주기
        enc_output, enc_hidden = encoder(inputs, enc_hidden)

        # 히든 스테이트 정의
        dec_hidden = copy.deepcopy(enc_hidden)
        dec_hidden2 = copy.deepcopy(enc_hidden)
        rl_dec_hidden = copy.deepcopy(enc_hidden)
        
        for t in range(inputs.shape[1]):
            
            '''
            SL 예측
            '''
            # SL 학습
            ## (SL-1) 다음 토큰을 예측하기
            preds, dec_hidden, _ = decoder(inputs[:, t], dec_hidden, enc_output)
            results = np.apply_along_axis(get_action, axis = 1, arr = np.array(preds), option = 'prob')
            predicted_token = tf.reshape(results[:, 0], shape = (-1, 1))        # SL 머신이 예측한 토큰
            
            ## (SL-2) 실제 타겟 데이터를 DB에서 받아오기
            tars = tf.reshape(targets[:, t], shape = (-1, 1))

            ## (SL-3) 손실 계산 : 매 t마다 손실 계산
            loss = ce_loss(tars, preds)

            ## (SL-4) 중요도 가중치 계산
            # pre_token = copy.deepcopy(inputs[:, t])
            # cur_token = copy.deepcopy(tars)
            # p_prob_vector = p_prob_mat[tf.cast(pre_token, dtype = tf.int64), tf.cast(cur_token, dtype = tf.int64)]
            # p_prob_cumul *= p_prob_vector

            # importance_ratio = tf.math.exp(tf.math.multiply(loss, -1)) / p_prob_vector # Importance Weight
            # importance_ratio = tf.math.exp(tf.math.multiply(loss, -1)) # Importance Weight
            # importance_weight *= importance_ratio

            ## (SL-3') 최종손실 계산 : 모든 t의 손실 합
            total_loss += loss

            ## 전체 sequence에 대해 구성 보상 계산 (position making)
            indicator = incidence_mat[np.array(predicted_token, dtype = int), t + 1]
            beta_score += indicator # 베타 점수

            '''
            시퀀스 수집
            '''
            ## (real & SL & RL) 실제 데이터 시퀀스 & SL이 예측한 시퀀스 누적
            real_seqs = np.append(real_seqs, tars, axis = 1)
            pred_seqs = np.append(pred_seqs, predicted_token, axis = 1)

        '''
        영양소 점수 계산
        '''
        # reward_shaping 여부 결정
        if reward_shaping == "False":
            # real sequence 전체에 대해 영양 점수 계산 
            nutrition_real = get_score_matrix(real_seqs, nutrient_data, food_dict)
            reward_real = np.apply_along_axis(get_reward_ver2, arr = nutrition_real, axis = 1, done = 0)
            reward_real = tf.cast(reward_real[:, 0].astype(float), dtype = tf.float32)
        else:
            # real sequence 전체에 대해 영양 점수 계산 
            nutrition_real = get_score_matrix(real_seqs, nutrient_data, food_dict)
            reward_real = np.apply_along_axis(get_reward_ver2, arr = nutrition_real, axis = 1, done = 0)
            reward_real = tf.cast(reward_real[:, 0].astype(float), dtype = tf.float32)
            
            # real sequence 전체에 대해 구성 점수 반영
            reward_real *= np.apply_along_axis(meal_hit_score, arr = real_seqs, axis = 1, category_data = category_data)[:, 0] * np.apply_along_axis(dish_hit_score, arr = real_seqs, axis = 1, category_data = category_data)[:, 0]

        # # pred sequence 전체에 대해 영양 점수 계산        
        # nutrition_pred = get_score_matrix(pred_seqs, nutrient_data, food_dict)
        # reward_pred = np.apply_along_axis(get_reward_ver2, arr = nutrition_pred, axis = 1, done = 0)
        # reward_pred = tf.cast(reward_pred[:, 0].astype(float), dtype = tf.float32)

        '''
        보상 계산
        '''
        ## (RL-6) RL 보상 계산
        # alergy관련 rewards 고려한다면
        if alergy_const == "True":
            alergic_rewards = is_alergy_trigger(real_seqs, alergy_menu_vector, food_dict)
            total_reward_pred = reward_real * (tf.reshape(beta_score, shape = (-1, )) / 15) * alergic_rewards

        # alergy관련 rewards 고려안한다면
        else:
            total_reward_pred = reward_real * (tf.reshape(beta_score, shape = (-1, )) / 15)

        ## reward_pred - reward_real을 통해 pred sequence의 보상이 real sequence의 보상보다 높아야만 양의 보상을 얻을 수 있음.
        # total_reward_pred = reward_pred * (tf.reshape(beta_score, shape = (-1, )) / 15)
        # total_reward_pred = (reward_pred - reward_real) * (tf.reshape(beta_score, shape = (-1, )) / 15)

        ## (RL-7) 손실 계산 (2) : 보상을 반영하여 최종 손실 계산 (Minimum Risk Training; MRT와 유사)
        # reward는 파라미터에 대한 함수가 아니므로 gradient가 전달되지 못하게 조치
        # final_loss = total_loss * total_reward_pred * (importance_weight + 1e-5)        # 활성화 - 비활성화 셋팅
        final_loss = total_loss * total_reward_pred

        '''
        Experience Buffer
        '''
        # Replay Buffer에 넣을 synthetic sequences 추출 (beta_score을 통과한 경우만)
        candidates_idx = np.where(beta_score == 15)[0]

        # 하나라도 candidate synthetic sequence가 있을 경우
        if len(candidates_idx) > 0: 
            buffer_memory = copy.deepcopy(np.array(x))
            experiences = pred_seqs[candidates_idx, :]       # synthetic sequence를 뽑아서
            buffer_memory[candidates_idx, :] = experiences      #   x의 candidate_idx 자리에 synthetic sequence를 buffer_memory 넣어준다.
        # candidate synthetic sequence가 전무할 경우
        else:
            buffer_memory = copy.deepcopy(np.array(x))  # 원래의 x를 그대로 buffer_memory로 활용


    '''
    SL 그라디언트 계산 및 업데이트 (학습)
    '''
    # (SL-3) SL 그라디언트 계산 및 업데이트
    grads = tape.gradient(final_loss, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))
    # rl_grads = rl_tape.gradient(rl_total_loss, RL_trainable_variables)
    # rl_optimizer.apply_gradients(zip(rl_grads, RL_trainable_variables))

    # 본 배치의 평균 손실
    batch_loss = (tf.reduce_mean(total_loss).numpy() / int(targets.shape[1]))
    # rl_batch_loss = (tf.reduce_mean(rl_total_loss).numpy() / int(targets.shape[1]))

    # return real_seqs, batch_loss, pred_seqs, rl_batch_loss, rl_pred_seqs, buffer_memory, _
    return real_seqs, batch_loss, pred_seqs, _, _, buffer_memory, _
# %%

def train_SEQ2SEQ_REINFORCE(x, encoder, c, c_encoder, decoder):
    # 토큰 시퀀스 배치 x와 초기화 된 encoder를 받는다.
    # 클러스터 시퀀스 배치 c와 초기화 된 c_encdoer를 받는다.
    # 초기화 된 decoer를 받는다

    # 손실함수, 옵티마이저 초기화
    ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction = 'none')
    sl_optimizer = tf.keras.optimizers.Adam(1e-3)

    # 두 인코더의 히든 스테이트 초기화
    enc_hidden = encoder.initialize_hidden_state()
    c_enc_hidden = encoder.initialize_hidden_state()

    # 토큰시퀀스 데이터 구축
    inputs = x[:, :x.shape[1] - 1]
    targets = x[:, 1:x.shape[1]]

    # 클러스터시퀀스 데이터
    c_inputs = c[:, :c.shape[1] - 1]

    # RL 에이전트가 예측하는 시퀀스 초기화
    rl_seqs = tf.reshape(inputs[:, 0], shape = (-1, 1))

    # SL 머신이 예측하는 시퀀스 초기화
    sl_pred_seqs = tf.reshape(inputs[:, 0], shape = (-1, 1))

    # SL은 Seq2Seq 파라미터를 업데이트
    SL_trainable_variables = encoder.trainable_variables + c_encoder.trainable_variables + decoder.trainable_variables

    # 각종 파라미터 초기화
    sl_loss = 0                                 # SL의 누적손실 초기화
    reward_over_timestep = 0                    # RL의 배치별 보상 초기화

    # total_score = 0
    # total_reward = 0                            # 총 보상 초기화

    '''
    Supervised Learning (SL) 파트
    '''
    with tf.GradientTape(watch_accessed_variables = False) as sl_tape:

        sl_tape.watch(encoder.trainable_variables)
        sl_tape.watch(c_encoder.trainable_variables)
        sl_tape.watch(decoder.trainable_variables)

        # 두 인코더의 컨텍스트 벡터 각각 뽑아주기
        enc_output, enc_hidden = encoder(inputs, enc_hidden)
        c_enc_output, c_enc_hidden = c_encoder(c_inputs, c_enc_hidden)


        # 두 인코더의 컨텍스트 벡터 연결해주기
        concat_hidden = tf.concat([enc_hidden, c_enc_hidden], axis = 1)

        # 연결된 컨텍스트 벡터를 sl과 rl 디코더의 초기 히든 스테이트로 정의
        sl_dec_hidden = copy.deepcopy(concat_hidden)
        rl_dec_hidden = copy.deepcopy(concat_hidden)

        for t in range(inputs.shape[1]):

            '''
            Reinforcement Learning (RL) 파트
            '''
            rl_dec_hidden, rl_seqs = train_REINFORCE(t, inputs, targets, rl_dec_hidden, rl_seqs, decoder)

            ## (RL-6) 배치 평균보상 계산 (생성 데이터 rl_seqs에 대해 계산)
            gen_nutrition_state = np.apply_along_axis(get_score_vector, arr = rl_seqs, axis = 1, nutrient_data = nutrient_data)
            gen_score = np.apply_along_axis(get_reward_ver2, arr = gen_nutrition_state, axis = 1, done = 0)
            # gen_reward = gen_score[:, 0] - gen_pre_score                  # 누적 보상
            # gen_pre_score = copy.deepcopy(gen_score[:, 0])                # pre_score 재할당
            gen_reward = gen_score[:, 0]                                    # 누적 보상
            mean_reward_over_data = np.mean(gen_reward)                     # 배치 평균보상 across data
            reward_over_timestep += mean_reward_over_data                   # 배치 평균보상 through timestep

            # SL 학습
            ## (SL-1) 예측
            sl_preds, sl_dec_hidden = decoder(inputs[:, t], sl_dec_hidden)
            results = np.apply_along_axis(get_action, axis = 1, arr = sl_preds, option = 'prob')
            predicted_token = tf.reshape(results[:, 0], shape = (-1, 1))        # SL 머신이 예측한 토큰

            ## (SL-2) 손실 계산 : 매 t마다 손실 계산
            tars = tf.reshape(targets[:, t], shape = (-1, 1))
            loss = ce_loss(tars, sl_preds)
            sl_loss += loss  # 누적 손실

            ## SL이 예측한 시퀀스 누적
            sl_pred_seqs = np.append(sl_pred_seqs, predicted_token, axis = 1)

        ## (RL-7) 배치 평균보상 최종 계산
        mean_reward_over_batch = reward_over_timestep / inputs.shape[1]

        # (SL-3) SL 그라디언트 계산 및 업데이트
        sl_grads = sl_tape.gradient(sl_loss, SL_trainable_variables)
        sl_optimizer.apply_gradients(zip(sl_grads, SL_trainable_variables))

        # 본 배치의 평균 손실
        batch_loss = (tf.reduce_mean(sl_loss).numpy() / int(targets.shape[1]))

    return batch_loss, mean_reward_over_batch, rl_seqs, sl_pred_seqs
def train_REINFORCE(t, inputs, targets, rl_dec_hidden, rl_seqs, decoder):
    '''
    t: current step
    inputs: inputs to be fed into RL agent
    targets: targets to be fed into RL agent in case we use off-policy control
    '''

    # 손실함수, 옵티마이저 초기화
    ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction = 'none')
    rl_optimizer = tf.keras.optimizers.Adam(1e-5)

    # 파라미터 초기화
    # pre_score = np.zeros([inputs.shape[0]])     # 이전 영양점수 초기화 (Behavioral Policy 용)
    # gen_pre_score = np.zeros([inputs.shape[0]]) # 이전 영양점수 초기화 (Target Policy 용)
    discounted_total_gain = 0
    discount = 0.99

    # RL은 Decoder 파라미터만 업데이트
    RL_trainable_variables = decoder.trainable_variables

    with tf.GradientTape(watch_accessed_variables = False) as rl_tape:
        rl_tape.watch(decoder.trainable_variables)

        # RL 학습
        ## (RL-1) 예측
        # rl_preds, rl_dec_hidden = decoder(rl_seqs[:, t], rl_dec_hidden)
        rl_preds, rl_dec_hidden = decoder(inputs[:, t], rl_dec_hidden)
        results = np.apply_along_axis(get_action, axis = 1, arr = rl_preds, option = 'prob')
        actions = tf.reshape(results[:, 0], shape = (-1, 1)) # 에이전트가 선택한 행위
        probs = tf.reshape(results[:, 1], shape = (-1, 1))  # 에이전트가 선택한 행위의 확률

        ## (RL-2) RL을 위한 시퀀스 누적
        rl_seqs = np.append(rl_seqs, actions, axis = 1)

        ## (RL-3) (Option 1) 매 t마다 보상 계산 (실제 데이터 targets에 대해 계산)
        # nutrition_state = np.apply_along_axis(get_score_vector, arr = rl_seqs, axis = 1, nutrient_data = nutrient_data)
        # nutrition_state = np.apply_along_axis(get_score_vector, arr = targets[:, :t], axis = 1, nutrient_data = nutrient_data)

        # Monte Carlo 샘플링 수행: t시점부터 ending point까지.   
        for m in range(inputs.shape[1] - t):
            nutrition_state = np.apply_along_axis(get_score_vector, arr = targets[:, :(t + m + 1)], axis = 1, nutrient_data = nutrient_data)
            score = np.apply_along_axis(get_reward_ver2, arr = nutrition_state, axis = 1, done = 0)
            reward = score[:, 0]  # 누적 보상
            # discounted_total_gain += (discount ** m) * reward   # 감가상각된 총 이득  (현재 지향 가중치)
            discounted_total_gain += (discount ** (inputs.shape[1] - t - m)) * reward   # 감가상각된 총 이득 (미래 지향 가중치)
            # print('discounted_total_gain :', discounted_total_gain)

            # reward = score[:, 0] - pre_score  # 누적 보상
            # pre_score = copy.deepcopy(score[:, 0]) # pre_score 재할당

        ## (RL-4) RL 손실 계산
        actions = tf.cast(actions, dtype = tf.int32) # 선택 행동 (자료형을 정수로 변환)
        # actions_vector = tf.squeeze(tf.one_hot(actions, depth = 1728))   # 원핫벡터로 인코딩 (We only transform into onehot vector when using CategoricalCrossentropy() not SparseCategoricalCrossentropy().)

        tars = targets[:, t]
        tars = tf.cast(tars, dtype = tf.int32)  # 정답 행동 (자료형을 정수로 변환)
        rl_loss = ce_loss(tars, rl_preds) * discounted_total_gain  # Deterministic Policy Gradient(DPG)

    ## (RL-5) RL 그라디언트 계산 및 업데이트
    rl_grads = rl_tape.gradient(rl_loss, RL_trainable_variables)
    rl_optimizer.apply_gradients(zip(rl_grads, RL_trainable_variables))

    '''
    rl_dec_hidden : RL 에이전트가 반환한 hidden_state
    rl_seqs : RL 에이전트의 trajectory
    '''
    return rl_dec_hidden, rl_seqs
# %%
