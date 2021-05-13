# %%
import sys
sys.path.append("/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code")
import os
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

from util import *
from Preprocessing import *
from Model import * 
# import argparse
# parser = argparse.ArgumentParser(description='gpu, num_epochs와 lr 입력')
# parser.add_argument('--num_epochs', type=int, required=True, help='num_epochs 입력')
# parser.add_argument('--lr', type=float, required=True, help='learning_rate 입력')
# parser.add_argument('--eb', type=int, required=True, help='epoch_of_buffer 입력')
# parser.add_argument('--bs', type=int, required=True, help='buffer_size 입력')
# parser.add_argument('--rs', type=str, required=True, help='reward_shaping 여부 (True or False)')
# parser.add_argument('--alergy', type=str, required=True, help='alergy 여부 (True or False)')
# args = parser.parse_args()
# print(args)

import numpy as np
import copy
import csv
import time
from multiprocessing import Pool

'''
Teacher-Forced REINFORCE (TFR)
'''
# 변수 초기화를 위한 random seed로서의 input, hidden_state, concat_state 생성
# Encoder to embed food sequence
encoder = Encoder(len(food_dict), BATCH_SIZE)
init_input = np.zeros([BATCH_SIZE, 1]) * 1726
init_hidden = encoder.initialize_hidden_state()
init_output, _ = encoder(init_input, init_hidden)

# Decoder to predict food sequence
decoder = Decoder(len(food_dict))
decoder(init_input, init_hidden, init_output)

# Parameter Initialization
# num_epochs = copy.deepcopy(args.num_epochs)
# lr = copy.deepcopy(args.lr)
# epoch_eb = copy.deepcopy(args.eb)
# target_buffer_size = copy.deepcopy(args.bs)
# reward_shaping = copy.deepcopy(args.rs)
# alergy_const = copy.deepcopy(args.alergy)

num_epochs = 5000
lr = 1e-3
epoch_eb = 5
target_buffer_size = 10
reward_shaping = False
alergy_const = False

buffer_idx = 0

# Parameter and Variable for Experience Forcing
experience_buffer = [ [[] for i in range(target_buffer_size)] for i in range(len(list(tf_dataset))) ]

# Define Checkpoint dir
## 일단 reward_shaping과 alergy를 같이 고려하는건 생각 안하기로.
# checkpoint_dir = "/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/Proposed/training_lr=" + str(lr) + "_epoch=" + str(num_epochs) + "_eb=" + str(epoch_eb) + "_bs=" + str(target_buffer_size) + "_rs=" + str(reward_shaping) + "_alergy=" + str(alergy_const) + "_TFR"

# createFolder(checkpoint_dir)
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(encoder = encoder, decoder = decoder)

per_epoch_rewards = []
start = time.time()  # 시작 시간 저장
for epoch in range(num_epochs):

    # rl_pred_seqs_all2 = np.empty((0, 16))
    full_batch_loss = 0
    rl_full_batch_loss = 0
    mean_rewards_per_batch = np.empty((0, 1))
    real_mean_rewards_per_batch = np.empty((0, 1))

    batch_num = 0
    # print('buffer_idx :', buffer_idx)

    for i in range(len(list(tf_dataset_update))):

        # 배치별로 시퀀스 값들 초기화
        real_seqs_all = np.empty((0, 16))
        pred_seqs_all = np.empty((0, 16))

        x = list(tf_dataset)[i]
        x_update = list(tf_dataset_update)[i]

        real_seqs, batch_loss, pred_seqs, _, _, buffer_memory, _ = train_SEQ2SEQ_TFR(x, x_update, encoder, decoder, lr, reward_shaping, alergy_const)
        real_seqs_all = np.append(real_seqs_all, real_seqs, axis = 0)  # 각 batch에서 생성한 결과물 누적합치기
        pred_seqs_all = np.append(pred_seqs_all, pred_seqs, axis = 0)  # 각 batch에서 생성한 결과물 누적합치기
        full_batch_loss += batch_loss

        # Experience Buffer에서 샘플링하여 tf_dataset 업데이트 하기
        ## (2) 각 배치의 (0번 buffer를 제외한) 나머지 buffer에는 experience로 채워주기 
        experience_buffer[batch_num][buffer_idx] = buffer_memory

        # batch_num 업데이트
        batch_num += 1

        # 배치별로 SL & RL의 생성 결과의 평균영양수준 확인
        # scores = np.apply_along_axis(get_score_vector, arr = pred_seqs_all, axis = 1, nutrient_data = nutrient_data)
        scores = get_score_matrix(pred_seqs_all, nutrient_data, food_dict)
        rewards = np.apply_along_axis(get_reward_ver2, arr = scores, axis = 1, done = 0)[:, 0]
        # mean_rewards = np.mean(rewards)
        # mean_rewards_per_batch = np.append(mean_rewards_per_batch, mean_rewards)

        # real_scores = get_score_matrix(real_seqs_all, nutrient_data, food_dict)
        # real_rewards = np.apply_along_axis(get_reward_ver2, arr = real_scores, axis = 1, done = 0)[:, 0]
        # real_mean_rewards = np.mean(real_rewards)
        # real_mean_rewards_per_batch = np.append(real_mean_rewards_per_batch, real_mean_rewards)


        # '''
        # 여기서 pred_seqs_all에 대해서 composition filter를 한번 통과시킨 애들에 대해서만 rewards를 계산해주면 됨
        # '''
        # compo_score = np.apply_along_axis(composition_score, axis = 1, arr = pred_seqs, category_data = category_data)
        # filtered_idx = np.where(compo_score >= 7)[0]
        # if len(filtered_idx) == 0:
        #     mean_rewards = 0
        # else:
        #     scores = np.apply_along_axis(get_score_vector, arr = pred_seqs_all[filtered_idx, :], axis = 1, nutrient_data = nutrient_data)
        #     rewards = np.apply_along_axis(get_reward_ver2, arr = scores, axis = 1, done = 0)[:, 0]
        #     mean_rewards = np.mean(rewards)

        # mean_rewards_per_batch = np.append(mean_rewards_per_batch, mean_rewards)

    per_epoch_rewards = rewards_matrix(epoch, rewards)
    # per_epoch_rewards = rewards_matrix(epoch, mean_rewards_per_batch)
    reward_df = pd.DataFrame(per_epoch_rewards)
    reward_df = reward_df.astype('float32')
    reward_df.columns = ['epoch', 'reward', 'sample']
    dir_file_name = save_reward_df(reward_df, "TFR", epoch_eb, target_buffer_size, lr, num_epochs)

    # 매 에포크 별 (즉, full_batch 기준) 손실
    epoch_loss = full_batch_loss / len(list(tf_dataset))

    if (epoch + 1) % 30 == 0:
        print('epoch : {}, epoch_loss : {}'.format(epoch, epoch_loss)) 
        print(' ')

        # 매 에포크 별 SL & RL 생성 결과 확인
        print('REAL 시퀀스 :', sequence_to_sentence(real_seqs_all, food_dict)[0])
        print('생성 시퀀스 :', sequence_to_sentence(pred_seqs_all, food_dict)[0])
        print('생성 시퀀스의 영양수준:', get_reward_ver2(get_score_vector(pred_seqs_all[0], nutrient_data), done = 0)[0])
        print(' ')

        # 매 에포크 별 생성 식단의 보상
        # print('REAL 시퀀스의 평균 영양수준 :', np.mean(real_mean_rewards_per_batch))
        print('생성 시퀀스의 평균 영양수준 :', np.mean(per_epoch_rewards))
        print(' ')
        print("total time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
    
    # 매 BUFFER_SIZE가 꽉 차면 buffer_idx 초기화
    if (buffer_idx + 1) % target_buffer_size == 0:
        buffer_idx = 0
    # buffer_idx 업데이트
    else:
        buffer_idx += 1

    # 매 epoch_eb 에포크 마다 experience_buffer를 활용해 tf_dataset 업데이트 해주기
    if (epoch + 1) % epoch_eb == 0:

        ## experience_buffer를 numpy 타입으로 변환
        eb_np = np.array(experience_buffer)
        
        ## 배치크기 (32, 16)와 동일한 empty numpy array 만들어주기
        # new_diet_data_np = np.empty([x.shape[0], x.shape[1]])
        new_diet_data_np = np.empty([0, x.shape[1]])

        ## 각 배치별로 experience_buffer에서 하나의 beffuer_memory를 샘플링한 후 new_diet_data_np에 stack하기
        for batch in range(eb_np.shape[0]):
            selected_buffer_idx = np.random.choice(eb_np[batch].shape[0])
            # selected_buffer_idx = np.random.choice(len(list(filter(None, eb_np[batch]))))
            # print('selected_buffer_idx', selected_buffer_idx)
            new_batch = eb_np[batch][selected_buffer_idx]
            new_diet_data_np = np.append(new_diet_data_np, new_batch, axis = 0)

        ## stack이 완료된 new_diet_data_np는 8-batch x 32-batch_size를 갖는 기존의 dataset과 동일한 크기의 data임
        ## new_diet_data_np를 tensor 객체로 바꿔준후 batch sliciing하기.
        tf_dataset_update = tf.data.Dataset.from_tensor_slices(new_diet_data_np)
        tf_dataset_update = tf_dataset_update.batch(BATCH_SIZE, drop_remainder = True)

    # # 체크포인트에 저장
    # if (epoch + 1) % 100 == 0:
    #     checkpoint.save(file_prefix = checkpoint_prefix)


# %%
