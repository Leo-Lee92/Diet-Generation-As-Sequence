# %%
from util import *
from main import get_params
args = get_params()
print('args :', args)

import tensorflow as tf
import numpy as np
import pandas as pd
import copy
import re

# Load required dataset
# Select language.
if args.language == 'english':
    ## -- (1) Load feature dataset
    feature_data = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data (new)/englishDB/new_nutrition.csv', encoding='CP949')

    ## -- (2) Load diet sequence dataset
    # if you want to load the diet data, including breafast meal, which is of 19 length.
    if args.add_breakfast == True:
        diet_data = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data (new)/englishDB/new_diet (breakfast added).csv', encoding='CP949')
        diet_data = diet_data.iloc[:, :-1]

    # if you want to load the diet data, without breakfast meal, which is of 14 length.
    else:                       
        diet_data = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data (new)/englishDB/new_diet (without breakfast).csv', encoding='CP949')

elif args.language == 'korean':
    ## -- (1) Load feature dataset
    feature_data = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data (new)/koreanDB/new_nutrition.csv', encoding='CP949')

    ## -- (2) Load diet sequence dataset
    # if you want to load the diet data, including breafast meal, which is of 19 length.
    if args.add_breakfast == True:
        diet_data = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data (new)/koreanDB/new_diet (breakfast added + non_spec-checker).csv', encoding='CP949')    # non-spec_checker
        diet_data = diet_data.iloc[:, :-1]
    # if you want to load the diet data, without breakfast meal, which is of 14 length.
    else:                       
        diet_data = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data (new)/koreanDB/new_diet (without breakfast).csv', encoding='CP949')
else:
    print('Error !. make clear which language of data to use.')

# Define parameters required for preprocessing
preprocessing_kwargs = {    
    'sequence_data' : diet_data,
    'feature_data' : feature_data,
    
    # Possible parameter list : ['or', 'arrange', 'correct1', 'correct2']
    'DB_quality' : 'correct2'
}

# Preprocess nutrition data
preprocess_nutrition = nutrition_preprocessor(**preprocessing_kwargs)
nutrient_data, food_dict = preprocess_nutrition()

# Preprocess diet data
preprocess_diets = diet_sequence_preprocessor(**preprocessing_kwargs)
diet_data = preprocess_diets(nutrient_data)

# Mapping from food to token
diet_data_np = food_to_token(diet_data, nutrient_data, empty_delete = True, num_empty = 3)  # delete diets whose the empty token appears more than 3 times.

# Mapping from diet_data_np to binary incidence format
incidence_data = diet_to_incidence(diet_data_np, food_dict)

# Define the size of batch. 
BATCH_SIZE = diet_data_np.shape[0]

# # Variable Parameter Initialization Ver.2 (If you pass the paremeters at Jupyter interactive level.)
# language = 'korean'
# add_breakfast = True
# network = 'GRU' 
# attention =  False 
# embed_dim = 128
# fc_dim = 64
# learning = 'off-policy'
# policy = 'random'
# beta = True
# buffer = True
# buffer_size = 5
# buffer_update = 5
# num_epochs = 10000
# lr = 1e-3
# kwargs = {
#     'language' : language,

#     # This parameter can be True or False.
#     # If it is set to be True, then the length of sequence is 21.
#     # On the other hand, if it is set to be False, the length of sequence is 16.
#     # The mode of get_reward_ver2() changes according to this parameter as well.
#     ## Possible parameter list = [True, False]
#     'add_breakfast' : add_breakfast,

#     # This parameter can be 'GRU' or 'LSTM'.
#     ## Possible parameter list = ['GRU', 'LSTM']
#     'fully-connected_layer' : network, 
    
#     # This parameter can be True or False. 
#     # If 'attention' = False, attention is not applied in the model.
#     ## Possible parameter list = [True, False]
#     'attention': attention, 

#     # This parameter represents the size of embedding layer (i.e., the number of neurons in the embedding layer).
#     ## Therefore, any integer value can be used for this parameter.
#     'embed_dim': embed_dim,

#     # This parameter represents the size of fully-connected layer (i.e., the number of neurons in the fully-connected layer).
#     ## Therefore, any integer value can be used for this parameter.
#     'fc_dim': fc_dim,

#     # This parameter can be 'on-policy' or 'off-policy'. 
#     # If 'policy_tpye' = 'on-policy', then target-policy and behavior policy is same (i.e., REINFORCE algorithm is run), which means actions are sampled according to target-policy distribution.
#     # If 'policy_type' = 'off-policy', then behavior policy is given as the stream of real data, which means actions are sampled according to data distribution.
#     ## Possible parameter list = ['on-policy', 'off-policy']
#     'learning': learning,

#     # This parameter can be random-policy, greedy-policy, and target-policy.
#     ## Possible paramter list = [random, greedy, target]
#     'policy': policy,

#     # This parameter can be True or False.
#     ## Possible parameter list = [True, False]
#     'use_beta': beta,

#     # This parameter can be True or False. 
#     # If 'use_buffer' = True, then the target data is replaced by synthetic data in stochastic way.
#     # On the other hand, if 'use_buffer' == False, then the target data is fixed in constant.
#     ## Possible parameter list = [True, False]
#     'use_buffer': buffer,

#     # Else parameters.
#     'buffer_size' : buffer_size,
#     'buffer_update' : buffer_update,
#     'num_epochs' : num_epochs,
#     'lr' : lr, 
#     'num_tokens' : len(food_dict),
#     'batch_size' : BATCH_SIZE
# }

# Define the argments which are passed to model and used in training procedure.
kwargs = {
    'language' : args.language,

    # This parameter can be True or False.
    # If it is set to be True, then the length of sequence is 21.
    # On the other hand, if it is set to be False, the length of sequence is 16.
    # The mode of get_reward_ver2() changes according to this parameter as well.
    ## Possible parameter list = [True, False]
    'add_breakfast' : args.add_breakfast,

    # This parameter can be 'GRU' or 'LSTM'.
    ## Possible parameter list = ['GRU', 'LSTM']
    'fully-connected_layer' : args.network, 
    
    # This parameter can be True or False. 
    # If 'attention' = False, attention is not applied in the model.
    ## Possible parameter list = [True, False]
    'attention': args.attention, 

    # This parameter represents the size of embedding layer (i.e., the number of neurons in the embedding layer).
    ## Therefore, any integer value can be used for this parameter.
    'embed_dim': args.embed_dim,

    # This parameter represents the size of fully-connected layer (i.e., the number of neurons in the fully-connected layer).
    ## Therefore, any integer value can be used for this parameter.
    'fc_dim': args.fc_dim,

    # This parameter can be 'on-policy' or 'off-policy'. 
    # If 'policy_tpye' = 'on-policy', then target-policy and behavior policy is same (i.e., REINFORCE algorithm is run), which means actions are sampled according to target-policy distribution.
    # If 'policy_type' = 'off-policy', then behavior policy is given as the stream of real data, which means actions are sampled according to data distribution.
    ## Possible parameter list = ['on-policy', 'off-policy']
    'learning': args.learning,

    # This parameter can be random-policy, greedy-policy, and target-policy.
    ## Possible paramter list = [random, greedy, target]
    'policy': args.policy,

    # This parameter can be True or False.
    ## Possible parameter list = [True, False]
    'use_beta': args.beta,

    # This parameter can be True or False. 
    # If 'use_buffer' = True, then the target data is replaced by synthetic data in stochastic way.
    # On the other hand, if 'use_buffer' == False, then the target data is fixed in constant.
    ## Possible parameter list = [True, False]
    'use_buffer': args.buffer,

    # Else parameters.
    'buffer_size' : args.buffer_size,
    'buffer_update' : args.buffer_update,
    'num_epochs' : args.num_epochs,
    'lr' : args.lr, 
    'num_tokens' : len(food_dict),
    'batch_size' : BATCH_SIZE
}

# Set buffer size of dataset to shuffle.
BUFFER_SIZE = int(BATCH_SIZE * (diet_data_np.shape[0] / BATCH_SIZE))

# Transform numpy object to tensorflow Dataset object and slices it into tensor-like form (i.e., define the number of dim and set the axis).
tf_dataset = tf.data.Dataset.from_tensor_slices(diet_data_np)

# Randomly shuffle the batch of BUFFER_SIZE and take it to define as tf_dataset
tf_dataset = tf_dataset.shuffle(buffer_size = BUFFER_SIZE, seed = 1234, reshuffle_each_iteration = False).take(BUFFER_SIZE)

# Make batch according to BATCH_SIZE.
tf_dataset = tf_dataset.batch(BATCH_SIZE, drop_remainder = True)

# Same as above but store the value to variable named tf_dataset_update.
tf_dataset_update = tf.data.Dataset.from_tensor_slices(diet_data_np)
tf_dataset_update = tf_dataset_update.shuffle(buffer_size = BUFFER_SIZE, seed = 1234, reshuffle_each_iteration = False).take(BUFFER_SIZE)
tf_dataset_update = tf_dataset_update.batch(BATCH_SIZE, drop_remainder = True)

# %%
