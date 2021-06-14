# %%
# Let's import instances (e.g., global variables, functions) from the following modules: {'Preprocessing', 'Model'}
## We inherits global variables, such as 'incidence_data', 'BATCH_SIZE' and 'food_dict', from Preprocessing.
## We inherits model instances (i.e., classes functionally related to model), such as 'decoder', 'encdoer' and 'sequence_generator', from Model.
from util import *
from Model import Sequence_Generator
from Preprocessing import food_dict, nutrient_data, incidence_data, tf_dataset, tf_dataset_update, kwargs

import tensorflow as tf
import numpy as np
import copy
import csv
import time

# Parameter (2) Constant Parameter Initialization
add_breakfast = kwargs['add_breakfast']
num_epochs = kwargs['num_epochs']
lr = kwargs['lr']
batch_size = kwargs['batch_size']
target_buffer_size = kwargs['buffer_size']
target_buffer_update = kwargs['buffer_update']

buffer_idx = 0
per_epoch_rewards = []
target_buffer = [ [[] for i in range(target_buffer_size)] for i in range(len(list(tf_dataset))) ]

# Make the directory that contains the results of training.
createFolder('/results')

'''
Teacher-Forced REINFORCE (TFR)
'''
# # Generate inital states for input, hidden, and concat state in Encoder and Decoder.
# ## --- (1) Define Encoder that embeds food sequences
# encoder = Encoder(len(food_dict), BATCH_SIZE, **kwargs)
# init_input = np.zeros([BATCH_SIZE, 1])
# init_hidden = encoder.initialize_hidden_state()
# init_output, _ = encoder(init_input, init_hidden)

# ## --- (2) Define Decoder that predicts food sequences
# decoder = Decoder(len(food_dict), **kwargs)
# decoder(init_input, init_hidden, init_output)

## --- (3) Define save_dir which represents the directory where Checkpoint is stored. The directory name consists of the parameters that controls the training of model.
root_dir = 'training_log/'
save_dir = createDir(root_dir, kwargs)

## --- (4) Check and save the parameters.
print(kwargs)
saveParams(save_dir + '/params', kwargs, food_dict, nutrient_data, incidence_data.numpy())


## --- (5) Define Checkpoint object
# Define checkpoint_prefix which is the prefix part of save_dir.
checkpoint_prefix = os.path.join(save_dir + '/checkpoints', "ckpt")

## --- (6) Define Generator object
diet_generator = Sequence_Generator(food_dict, nutrient_data, incidence_data, **kwargs)


# Define checkpoint object in the variable 'checkpoint'.
# checkpoint = tf.train.Checkpoint(encoder = encoder, decoder = decoder, kwargs = kwargs)
checkpoint = tf.train.Checkpoint(generator = diet_generator, params = kwargs)

## --- (5) Train the model.
start = time.time() # start time
for epoch in range(num_epochs):

    # Initialize total cumulative loss
    full_batch_loss = 0

    # Initialize per batch mean reward.
    mean_rewards_per_batch = np.empty((0, 1))

    # Initialize the batch number.
    batch_num = 0

    # Do training
    for i in range(len(list(tf_dataset_update))):

        # Define i-th batch of original and (potential) update dataset.
        x = list(tf_dataset)[i]
        x_update = list(tf_dataset_update)[i]

        # Initialize (diet) sequences at every batch
        ## full_seq_len is the length of full sequence.
        full_seq_len = x.shape[1]
        real_seqs_all = np.empty((0, full_seq_len))
        pred_seqs_all = np.empty((0, full_seq_len))

        # Train Teacher-Forced REINFORCe (TFR)
        real_seqs, batch_loss, pred_seqs, _, _, synthetic_target, _ = diet_generator.train(x, x_update)
        real_seqs_all = np.append(real_seqs_all, real_seqs, axis = 0)  # stack real diet sequences from each batch
        pred_seqs_all = np.append(pred_seqs_all, pred_seqs, axis = 0)  # stack synthetic diet sequences generated based on each batch
        full_batch_loss += batch_loss

        # Fill it target_buffer with synthetic target at every batch and buffer according to batch_num and buffer_idx.
        target_buffer[batch_num][buffer_idx] = synthetic_target

        # Update batch_num
        batch_num += 1

        # (Full batch) Store nutrition scores and rewards of synthetic diets generated at each batch.
        scores = get_score_matrix(pred_seqs_all, food_dict, nutrient_data)
        rewards = np.apply_along_axis(get_reward_ver2, arr = scores, axis = 1, done = 0, mode = add_breakfast)[:, 0]

        # (Batch) Store nutrition scores and rewards of synthetic diets generated at each batch.
        # mean_rewards = np.mean(rewards)
        # mean_rewards_per_batch = np.append(mean_rewards_per_batch, mean_rewards)
    
    # Reset the value of buffer_idx as 0 when the buffer_idx gets eqaul to target_buffer_size.
    if (buffer_idx + 1) % target_buffer_size == 0:
        buffer_idx = 0

    # Move to the next buffer by increasing 'buffer_idx'.
    else:
        buffer_idx += 1

    # Update on-training dataset using target_buffer which is composed of synthetic diets.
    tf_dataset_update  = update_dataset(epoch, batch_size, target_buffer_update, target_buffer, tf_dataset_update, x, food_dict, nutrient_data, kwargs['add_breakfast'])

    # Store the checkpoint.
    if (epoch + 1) % 100 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    # (Full batch) Define per_epoch_rewards variable using rewards_matrix function.
    per_epoch_rewards = rewards_matrix(epoch, rewards)

    # (Batch) Define per_epoch_rewards variable using rewards_matrix function.
    # per_epoch_rewards = rewards_matrix(epoch, mean_rewards_per_batch)

    # Save the reward per epoch
    reward_df = pd.DataFrame(per_epoch_rewards)
    reward_df = reward_df.astype('float32')
    reward_df.columns = ['epoch', 'reward', 'sample']
    dir_file_name = save_reward_df(reward_df, "TFR", target_buffer_update, target_buffer_size, lr, num_epochs)

    # Compute loss per epoch
    epoch_loss = full_batch_loss / len(list(tf_dataset))

    if (epoch + 1) % 100 == 0:
        print('epoch : {}, epoch_loss : {}'.format(epoch, epoch_loss)) 
        print(' ')

        # 매 에포크 별 SL & RL 생성 결과 확인
        print('REAL 시퀀스 :', sequence_to_sentence(real_seqs_all, food_dict)[0])
        print('REAL 시퀀스의 영양수준:', get_reward_ver2(get_score_vector(real_seqs_all[0], nutrient_data), done = 0, mode = add_breakfast)[0])
        print(' ')
        print('생성 시퀀스 :', sequence_to_sentence(pred_seqs_all, food_dict)[0])
        print('생성 시퀀스의 영양수준:', get_reward_ver2(get_score_vector(pred_seqs_all[0], nutrient_data), done = 0, mode = add_breakfast)[0])
        print(' ')

        # Calculate the (nutrient) score of the real and generated diets.
        nutrient_real = np.apply_along_axis(get_score_vector, axis = 1, arr = np.array(real_seqs_all), nutrient_data = nutrient_data)
        nutrient_gen = np.apply_along_axis(get_score_vector, axis = 1, arr = pred_seqs_all, nutrient_data = nutrient_data)

        # Get reward-related information of true and generated diet sequences.
        reward_info_real = np.apply_along_axis(get_reward_ver2, axis = 1, arr = nutrient_real, done = 0, mode = kwargs['add_breakfast'])
        reward_info_gen = np.apply_along_axis(get_reward_ver2, axis = 1, arr = nutrient_gen, done = 0, mode = kwargs['add_breakfast'])

        # Calculate mean rewards of true and generated diet sequences.
        mean_true_reward = np.mean(reward_info_real[:, 0])
        mean_gen_reward = np.mean(reward_info_gen[:, 0])
        print('REAL 평균 보상 :', mean_true_reward)
        print('생성 평균 보상 :', mean_gen_reward)

        print(' ')
        print("total time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

# %%
