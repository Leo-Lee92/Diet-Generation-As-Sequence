# %%
import os
import sys
sys.path.append("/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code")
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_virtual_device_configuration(gpu,
      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 2048)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import numpy as np
import copy
import csv
import time
sys.path.append("/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code")
from Preprocessing import *
from util import *
from Model import * 

import argparse
parser = argparse.ArgumentParser(description='num_epochs와 lr 입력')
parser.add_argument('--num_epochs', type=int, required=True, help='num_epochs 입력')
parser.add_argument('--lr', type=float, required=True, help='learning_rate 입력')
parser.add_argument('--rs', type=str, required=True, help='reward_shaping 여부 (True or False)')
args = parser.parse_args()

'''
MIXER 학습
'''
# 변수 초기화를 위한 random seed로서의 input, hidden_state, concat_state 생성
encoder = Encoder(len(food_dict), BATCH_SIZE)
init_input = np.zeros([BATCH_SIZE, 1])
init_hidden = encoder.initialize_hidden_state()
init_output, _ = encoder(init_input, init_hidden)

# Decoder to predict food sequence
decoder = Decoder(len(food_dict))
decoder(init_input, init_hidden, init_output)

# 파라미터 초기화
num_epochs = copy.deepcopy(args.num_epochs)
lr = copy.deepcopy(args.lr)
reward_shaping = copy.deepcopy(args.rs)

# num_epochs = 5000
# lr = 1e-3

len_seq = diet_data_np.shape[1] - 1 - 1      # 첫번째 -1 : length -> index / 두번째 -1 : action 고려
curriculum_step = 0
per_epoch_rewards = []


# Define Checkpoint dir - BASELINE-RL을 학습한 결과 체크포인트 저장
checkpoint_dir = "/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/Baselines/MIXER/training_lr=" + str(lr) + "_epoch=" + str(num_epochs) + "_rs=" + str(reward_shaping) + "_MIXER"

createFolder(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(encoder = encoder, decoder = decoder)

start = time.time()  # 시작 시간 저장
for epoch in range(num_epochs):
    full_batch_loss = 0
    pred_seqs_all = np.empty((0, 16))
    train_seqs_all = np.empty((0, 16))
    pred_mean_rewards_per_batch = np.empty((0, 1))
    
    curriculum_range, curriculum_step = do_curriculum(True, epoch, num_epochs, len_seq, curriculum_step)

    '''
    Reinforcement Learning with Curriculum Learning
    '''
    for x in tf_dataset:
        # x : token sequence

        batch_loss, pred_seqs, train_seqs = train_MIXER(x, encoder, decoder, curriculum_range, lr, reward_shaping)    # sampler: decoder 또는 pretrain_decoder
        pred_seqs_all = np.append(pred_seqs_all, pred_seqs, axis = 0)  # 각 batch에서 SL이 생성한 결과물 누적합치기
        train_seqs_all = np.append(train_seqs_all, train_seqs, axis = 0)  # 각 batch에서 SL이 생성한 결과물 누적합치기
        full_batch_loss += batch_loss

        # 배치별로 생성 시퀀스의 평균영양수준 확인
        ## 생성 시퀀스
        # pred_scores = np.apply_along_axis(get_score_vector, arr = pred_seqs_all, axis = 1, nutrient_data = nutrient_data)
        pred_scores = get_score_matrix(pred_seqs_all, nutrient_data, food_dict)
        pred_rewards = np.apply_along_axis(get_reward_ver2, arr = pred_scores, axis = 1, done = 0)[:, 0]
        pred_mean_rewards = np.mean(pred_rewards)
        pred_mean_rewards_per_batch = np.append(pred_mean_rewards_per_batch, pred_mean_rewards)

    per_epoch_rewards = rewards_matrix(epoch, pred_rewards)
    reward_df = pd.DataFrame(per_epoch_rewards)
    reward_df = reward_df.astype('float32')
    reward_df.columns = ['epoch', 'reward', 'sample']
    dir_file_name = save_reward_df(reward_df, "MIXER", None, None, lr, num_epochs)

    # 매 에포크 별 (즉, full_batch 기준) 손실
    epoch_loss = full_batch_loss / len(list(tf_dataset))    
    if (epoch + 1) % 30 == 0:
        print('epoch : {}, curriculum_range : {}, epoch_loss : {}'.format(epoch, curriculum_range, epoch_loss)) 

        # 매 에포크 별 생성 결과 확인
        print('REAL 시퀀스 (target + pred) :', sequence_to_sentence(train_seqs_all, food_dict)[0])
        print('REAL 시퀀스 (target + pred) :', get_reward_ver2(get_score_vector(train_seqs_all[0], nutrient_data), done = 0)[0])
        print(' ')
        print('생성 시퀀스 (pred_seqs) :', sequence_to_sentence(pred_seqs_all, food_dict)[0])
        print('생성 시퀀스 (pred_seqs)의 영양수준:', get_reward_ver2(get_score_vector(pred_seqs_all[0], nutrient_data), done = 0)[0])
        print(' ')

        # 매 에포크 별 배치전체 기준 SL의 생성 결과의 평균영양수준 확인
        # nutrition_scores = np.apply_along_axis(get_score_vector, arr = pred_seqs_all, axis = 1, nutrient_data = nutrient_data)
        nutrition_scores = get_score_matrix(pred_seqs_all, nutrient_data, food_dict)
        rewards = np.apply_along_axis(get_reward_ver2, arr = nutrition_scores, axis = 1, done = 0)[:, 0]
        mean_reward = np.mean(rewards)
        print('생성 시퀀스의 평균영양수준:', mean_reward)
        print(' ')
        print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

    # 체크포인트에 저장
    if (epoch + 1) % 100 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
