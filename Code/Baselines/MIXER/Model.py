# %%
from util import *
from Preprocessing import *
import tensorflow as tf
import numpy as np
import copy
import pandas as pd
import sys
sys.path.append("/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code")

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

# def train_SEQ2SEQ(x, encoder, decoder):
#     # 토큰 시퀀스 배치 x와 초기화 된 encoder를 받는다.
#     # 초기화 된 decoer를 받는다

#     # 손실함수, 옵티마이저 초기화
#     ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction = 'none')
#     sl_optimizer = tf.keras.optimizers.Adam(1e-3)

#     # 인코더의 히든 스테이트 초기화
#     enc_hidden = encoder.initialize_hidden_state()

#     # 토큰시퀀스 데이터 구축
#     inputs = x[:, :x.shape[1] - 1]
#     targets = x[:, 1:x.shape[1]]

#     # RL 에이전트가 예측하는 시퀀스 초기화
#     rl_seqs = tf.reshape(inputs[:, 0], shape = (-1, 1))

#     # SL 머신이 예측하는 시퀀스 초기화
#     sl_pred_seqs = tf.reshape(inputs[:, 0], shape = (-1, 1))

#     # SL은 Seq2Seq 파라미터를 업데이트
#     SL_trainable_variables = encoder.trainable_variables + decoder.trainable_variables

#     # 각종 파라미터 초기화
#     sl_loss = 0                           # SL의 누적손실 초기화
    
#     '''
#     Supervised Learning (SL) 파트
#     '''
#     with tf.GradientTape(watch_accessed_variables = False) as sl_tape:
#         sl_tape.watch(SL_trainable_variables)

#         # 인코더의 컨텍스트 벡터 뽑아주기
#         enc_output, enc_hidden = encoder(inputs, enc_hidden)

#         # 히든 스테이트 정의
#         sl_dec_hidden = copy.deepcopy(enc_hidden)

#         for t in range(inputs.shape[1]):
            
#             '''
#             SL 학습
#             '''
#             # SL 학습
#             ## (SL-1) 예측
#             sl_preds, sl_dec_hidden, attention_weights = decoder(inputs[:, t], sl_dec_hidden, enc_output)
#             results = np.apply_along_axis(get_action, axis = 1, arr = sl_preds, option = 'prob')
#             predicted_token = tf.reshape(results[:, 0], shape = (-1, 1))        # SL 머신이 예측한 토큰
#             # results = get_action(np.array(sl_preds), option = 'prob')
#             # predicted_token = tf.reshape(results[0], shape = (-1, 1))        # SL 머신이 예측한 토큰

#             ## (SL-2) 손실 계산 : 매 t마다 손실 계산
#             tars = tf.reshape(targets[:, t], shape = (-1, 1))
#             loss = ce_loss(tars, sl_preds)
#             sl_loss += loss  # 누적 손실

#             ## SL이 예측한 시퀀스 누적
#             sl_pred_seqs = np.append(sl_pred_seqs, predicted_token, axis = 1)

#         # (SL-3) SL 그라디언트 계산 및 업데이트
#         sl_grads = sl_tape.gradient(sl_loss, SL_trainable_variables)
#         sl_optimizer.apply_gradients(zip(sl_grads, SL_trainable_variables))

#         # 본 배치의 평균 손실
#         batch_loss = (tf.reduce_mean(sl_loss).numpy() / int(targets.shape[1]))

#     return batch_loss, sl_pred_seqs

def train_MIXER(x, encoder, decoder, curriculum_range, lr, reward_shaping = False):

    '''
    t: current step
    inputs: inputs to be fed into RL agent
    targets: targets to be fed into RL agent in case we use off-policy control
    '''
    # 손실함수, 옵티마이저 초기화
    ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction = 'none')

    if curriculum_range == None:
        MIXER_optimizer = tf.keras.optimizers.Adam(lr)
    else:
        MIXER_optimizer = tf.keras.optimizers.Adam(lr)


    # 두 인코더의 히든 스테이트 초기화
    enc_hidden = encoder.initialize_hidden_state()

    # 토큰시퀀스 데이터 구축
    inputs = x[:, :x.shape[1] - 1]
    targets = x[:, 1:x.shape[1]]

    # RL 에이전트가 예측하는 시퀀스 초기화
    pred_seqs = tf.reshape(inputs[:, 0], shape = (-1, 1))       # 예측한 시퀀스
    train_seqs = tf.reshape(inputs[:, 0], shape = (-1, 1))      # 학즙한 시퀀스 (타겟 + 예측)

    # 각종 파라미터 초기화
    MIXER_loss = 0                                 # SL의 누적손실 초기화
    reward_over_timestep = 0                    # RL의 배치별 보상 초기화

    # 파라미터 초기화
    # pre_score = np.zeros([inputs.shape[0]])     # 이전 영양점수 초기화 (Behavioral Policy 용)
    # gen_pre_score = np.zeros([inputs.shape[0]]) # 이전 영양점수 초기화 (Target Policy 용)

    # 업데이트 파라미터 지정
    MIXER_trainable_variables = decoder.trainable_variables

    # 손실 초기화
    full_MIXER_loss = 0

    with tf.GradientTape(watch_accessed_variables = False) as MIXER_tape:
        MIXER_tape.watch(MIXER_trainable_variables)
    
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
                MIXER_loss = ce_loss(targets[:, t], preds)

            # (2) 만약 curriculumn learning 하지 않는다면 -> 바로 REINFORCE Learning 하라
            ## 주의. 정상적인 sequence가 나오려면 반드시 encoder-decoder가 사전학습 되어 있어야 함.
            elif curriculum_range == None:
                # RL 학습
                # 예측
                preds, dec_hidden, _ = decoder(train_seqs[:, t], dec_hidden, enc_output)                             # 자기 자신 (pred_seqs)을 인풋으로 받음
                results = np.apply_along_axis(get_action, axis = 1, arr = preds, option = 'prob')
                actions = tf.reshape(results[:, 0], shape = (-1, 1))                                        # 에이전트가 선택한 행위
                probs = tf.reshape(results[:, 1], shape = (-1, 1))                                          # 에이전트가 선택한 행위의 확률

                # 시퀀스 누적
                pred_seqs = np.append(pred_seqs, actions, axis = 1)
                train_seqs = np.append(train_seqs, actions, axis = 1)                                             # curriculum_learning 중일 때 curriculum_range 이후엔 예측 데이터 (actions)를 입력 

                # RL 손실 계산
                MIXER_loss = ce_loss(actions, preds)       # REINFORCE 손실 계산                                   

            # (3) 만약 curriculumn learning 시작했는데
            else:
                # 현재 step이 curriculum_range 안에 있을 경우 (현 step은 curriculum learning 해야한다면) -> XE_gradient로 Learning 하라
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
                    MIXER_loss = ce_loss(targets[:, t], preds)                                             
                    # print('curriculum_range : {}, epoch : {}, t : {}, XENT Loss : {}'.format(curriculum_range, epoch, t, tf.reduce_mean(MIXER_loss)))
                
                # 현재 step이 curriculum_range 밖에 있을 경우 (현 step에선 curriculum learning 해야 한다면) -> policy_gradient로 Learning 하라
                else:   
                    # RL 학습
                    # 예측
                    preds, dec_hidden, _ = decoder(train_seqs[:, t], dec_hidden, enc_output)                             # 자기 자신 (pred_seqs)을 인풋으로 받음
                    results = np.apply_along_axis(get_action, axis = 1, arr = preds, option = 'prob')
                    actions = tf.reshape(results[:, 0], shape = (-1, 1))                                        # 에이전트가 선택한 행위
                    probs = tf.reshape(results[:, 1], shape = (-1, 1))                                          # 에이전트가 선택한 행위의 확률

                    # 시퀀스 누적
                    pred_seqs = np.append(pred_seqs, actions, axis = 1)
                    train_seqs = np.append(train_seqs, actions, axis = 1)                                             # curriculum_learning 중일 때 curriculum_range 이후엔 예측 데이터 (actions)를 입력 

                    # RL 손실 계산
                    MIXER_loss = ce_loss(actions, preds)       # REINFORCE 손실 계산

            full_MIXER_loss += MIXER_loss
            
        #  손실에 보상 반영여부 결정
        if curriculum_range != "All":    # 만약 curriculum learning이 더이상 없다면 보상을 반영한 손실 (XE+R)
            # nutrition_state = np.apply_along_axis(get_score_vector, arr = train_seqs, axis = 1, nutrient_data = nutrient_data)
            nutrition_state = get_score_matrix(train_seqs, nutrient_data, food_dict)
            score = np.apply_along_axis(get_reward_ver2, arr = nutrition_state, axis = 1, done = 0)[:, 0]
            if reward_shaping == "False":
                reward = score          # 보상
                reward = reward.reshape(-1, 1)
            else:
                # 쉐이핑 된 보상
                reward = score * np.apply_along_axis(meal_hit_score, arr = train_seqs, axis = 1, category_data = category_data)[:, 0] * np.apply_along_axis(dish_hit_score, arr = train_seqs, axis = 1, category_data = category_data)[:, 0]
                reward = reward.reshape(-1, 1)

            final_loss = full_MIXER_loss * reward 

        else:   # 만약 curriculum learning이 All이라면 순전히 XENT만 고려한 손실 (XENT)
            final_loss = full_MIXER_loss     

    ## (RL-5) RL 그라디언트 계산 및 업데이트
    MIXER_grads = MIXER_tape.gradient(final_loss, MIXER_trainable_variables)
    MIXER_optimizer.apply_gradients(zip(MIXER_grads, MIXER_trainable_variables))

    # 본 배치의 평균 손실
    batch_loss = (tf.reduce_mean(full_MIXER_loss).numpy() / int(inputs.shape[1]))

    '''
    batch_loss : RL 에이전트의 배치별 평균 손실
    pred_seqs : RL 에이전트가 예측한 trajectory
    train_seqs : RL 에이전트가 실제로 학습한 trajectory
    '''
    return batch_loss, pred_seqs, train_seqs


def MC_simulation(x, t, inputs, rl_seqs, dec_hidden, samplar):
    '''
    x: 현재 배치
    t : 현재 스텝 (= MC 시뮬레이션을 시작해야할 스텝은 t + 1)
    inputs : 전체 시퀀스의 길이를 받아오기 위해 필요한 인자
    rl_seqs : 에이전트가 현재까지 생성한 결과물
    dec_hidden : 에이전트의 현재 히든 스테이트
    pretrain_decoder: 사전학습 시켜놓은 decoder. (decoder는 필요없음)
    '''
    x = np.array(x)
    incidence_mat = np.array(diet_to_incidence(x, food_dict))
    incidence_mat[incidence_mat > 0] = 1

    ## (RL-3) (Option 1) 매 t마다 보상 계산 (실제 데이터 targets에 대해 계산) - Monte Carlo 샘플링 수행: t시점부터 ending point까지. 
    # MC 시뮬레이션 파라미터 초기화
    mc_seqs = copy.deepcopy(rl_seqs)                # monte-carlo 시퀀스 초기화
    mc_dec_hidden = copy.deepcopy(dec_hidden)    # monte-carlo 히든 스테이트 초기화
    discounted_total_gain = 0
    position_mask_overlap = np.ones([inputs.shape[0], 1])
    discount = 0.99

    '''
    - inputs.shape[1] - 1 인 이유는, shape는 1부터 시작하는 '길이'를 의미하고 't'는 0부터 시작하는 '인덱스'이므로
    - '인덱스' 기준으로 숫자를 맞춰주기 위해 - 1 해주었음 
    '''
    # 만약 현 step t가 마지막 step이 아니라면
    if t < (inputs.shape[1] - 1):
        '''
        - action에 대한 평가를 하기 위해 MC를 진행하는 것이므로, action에 의해 '한 단계 더 진행'된 상태에서 시작하는 것임. 
        - '한 단계 더'를 반영하여, 전체 (시퀀스 길이 -1) 인 길이까지만 loop 해주기.
        '''

        # MC 시뮬레이션 수행
        for m in range(inputs.shape[1] - 1 - t):
            # MC 시뮬레이션을 통한 mc_action 샘플링
            # t + 1은, MC 시뮬레이션이 t에서 action이 추가된 상태에서 시작됨을 반영한 것
            tars = tf.reshape(mc_seqs[:, (t + 1 + m)], shape = (-1, 1))

            mc_preds, mc_dec_hidden = samplar(tars, mc_dec_hidden)                      # samplar가 decoder일 떄
            # mc_preds = RollOutSamplar(incidence_data, (t + 1 + m), inputs)            # random samplar
            mc_results = np.apply_along_axis(get_action, axis = 1, arr = np.array(mc_preds), option = 'prob')
            mc_actions = tf.reshape(mc_results[:, 0], shape = (-1, 1))
            mc_probs = tf.reshape(mc_results[:, 1], shape = (-1, 1))

            # '''
            # 제대로 MC 시뮬레이션 되는지 확인
            # '''
            # print('rl_seqs :', sequence_to_sentence(rl_seqs, food_dict)[0])
            # print('{}-token, tars : {}, {} -th, mc_actions : {}'.format(t, sequence_to_sentence(tars.numpy(), food_dict)[0], m, sequence_to_sentence(mc_actions.numpy(), food_dict)[0]))
            # print(' ')

            # 샘플링된 mc_action을 mc_seqs에 추가 -> 현재 자신의 정책을 활용해 예측한 샘플들로부터 보상을 추정하기 위함
            mc_seqs = np.append(mc_seqs, mc_actions, axis = 1)

            # 구성보상
            position_mask = incidence_mat[np.array(tars, dtype = int), (t + 1 + m)]
            position_mask_overlap *= position_mask

            # 영양보상 추정
            '''
            - t + 1 + m 에서 첫번째 + 1은, MC 시뮬레이션 수행전 t스텝에서 선택한 action이 야기한 보상도 포함하여 보상을 계산해야 함으로 + 1해주었음.
            - 두번쨰 + 1은 마지막 인덱스를 배제하는 ':'의 기능적 특성을 고려하여 + 1 해주었음 
            '''
            nutrition_state = np.apply_along_axis(get_score_vector, arr = mc_seqs[:, :(t + 1 + m + 1)], axis = 1, nutrient_data = nutrient_data)
            score = np.apply_along_axis(get_reward_ver2, arr = nutrition_state, axis = 1, done = 0)
            reward = score[:, 0]  # 누적 보상
            reward = reward.reshape((inputs.shape[0], -1))

            discounted_total_gain += (discount ** m) * position_mask * reward                               # 감가상각된 총 이득  (현재 지향 가중치)
            # discounted_total_gain += (discount ** (inputs.shape[1] - t - m)) * position_mask * reward     # 감가상각된 총 이득 (미래 지향 가중치)
            
            # reward = score[:, 0] - pre_score  # 누적 보상
            # pre_score = copy.deepcopy(score[:, 0]) # pre_score 재할당

    # 만약 현 step t가 마지막 step이라면
    else:
        tars = tf.reshape(mc_seqs[:, (t + 1)], shape = (-1, 1))
        # 구성보상
        position_mask = incidence_mat[np.array(tars, dtype = int), (t + 1)]
        position_mask_overlap *= position_mask

        # 영양보상 추정
        '''
        - t + 1 에서 + 1은, MC 시뮬레이션 수행전 t스텝에서 선택한 action이 야기한 보상도 포함하여 보상을 계산해야 함으로 + 1해주었음.
        - 두번쨰 + 1은 마지막 인덱스를 배제하는 ':'의 기능적 특성을 고려하여 + 1 해주었음
        '''
        nutrition_state = np.apply_along_axis(get_score_vector, arr = mc_seqs[:, :(t + 1 + 1)], axis = 1, nutrient_data = nutrient_data)
        score = np.apply_along_axis(get_reward_ver2, arr = nutrition_state, axis = 1, done = 0)
        reward = score[:, 0]  # 누적 보상
        reward = reward.reshape((inputs.shape[0], -1))

        discounted_total_gain += (discount ** (inputs.shape[1] - t - 1)) * position_mask * reward       # 감가상각된 총 이득  (현재 지향 가중치)
        # discounted_total_gain += (discount ** (inputs.shape[1] - t - 1)) * position_mask * reward     # 감가상각된 총 이득 (미래 지향 가중치)
        # reward = score[:, 0] - pre_score  # 누적 보상
        # pre_score = copy.deepcopy(score[:, 0]) # pre_score 재할당


    discounted_total_gain = tf.convert_to_tensor(discounted_total_gain, dtype = tf.float32)

    return discounted_total_gain
# %%
