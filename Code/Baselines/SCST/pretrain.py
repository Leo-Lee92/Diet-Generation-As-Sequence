# %%
import os
import sys
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

'''
사전학습 하기
'''
# Encoder to embed food sequence
encoder = Encoder(len(food_dict), BATCH_SIZE)

# Decoder to predict food sequence
decoder = Decoder(len(food_dict))

# Define Checkpoint dir
checkpoint_dir = "/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/Baselines/SCST/pretraining_SCST"
createFolder(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(encoder = encoder, decoder = decoder)

init_lr = 1 * 1e-3
lr = copy.deepcopy(init_lr)
ss_prob = 5 * 1e-5

start = time.time()  # 시작 시간 저장
epoch = 0
while ss_prob < 0.25:
    epoch += 1    
    xe_full_batch_loss = 0
    xe_pred_seqs_all = np.empty((0, 16))

    for x in tf_dataset:

        xe_batch_loss, xe_pred_seqs = train_SEQ2SEQ_for_SCST(x, encoder, decoder, epoch, lr, ss_prob)
        xe_pred_seqs_all = np.append(xe_pred_seqs_all, xe_pred_seqs, axis = 0)  # 각 batch에서 SL이 생성한 결과물 누적합치기
        xe_full_batch_loss += xe_batch_loss
    
    # 매 에포크 별 (즉, full_batch 기준) 손실
    epoch_loss = xe_full_batch_loss / len(list(tf_dataset))    
    print('epoch : {}, epoch_loss : {}, lr : {}, ss_prob : {}'.format(epoch, epoch_loss, lr, ss_prob)) 

    # 매 에포크 별 생성 결과 확인
    print('XE-loss 생성결과 :', sequence_to_sentence(xe_pred_seqs_all, food_dict)[0])
    print('XE-loss 생성결과의 영양수준:', get_reward_ver2(get_score_vector(xe_pred_seqs_all[0], nutrient_data), done = 0)[0])
    print(' ')
    print("time :", time.time() - start)

    # # lr 담금질
    # if (epoch + 1) % 100 == 0:
    #     lr *= 0.95

    # predicted_token을 target으로 활용할 확률 스케쥴링.
    if (epoch + 1) % 10 == 0:
        ss_prob = ss_prob + (ss_prob * 0.05)        

    # 체크포인트에 저장
    if (epoch + 1) % 100 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
