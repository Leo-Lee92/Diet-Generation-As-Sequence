# %%
from util import *
import tensorflow as tf
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.cluster import AffinityPropagation

## 영양소 data
raw_data = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data/foods2.csv', encoding='CP949')
# nutrient_data = raw_data.iloc(axis = 1)[:23]

empty_vector = pd.DataFrame(0, columns = raw_data.columns, index = [0])
empty_vector['name'] = "empty"
empty_vector['group'] = 8
raw_data = pd.concat([empty_vector, raw_data]).reset_index(drop = True)

start_vector = pd.DataFrame(0, columns = raw_data.columns, index = [0])
start_vector['name'] = "시작"
start_vector['group'] = 9
raw_data = pd.concat([raw_data, start_vector]).reset_index(drop = True)

end_vector = pd.DataFrame(0, columns = raw_data.columns, index = [0])
end_vector['name'] = "종료"
end_vector['group'] = 9
raw_data = pd.concat([raw_data, end_vector]).reset_index(drop = True)

nutrient_data = raw_data.iloc(axis = 1)[:22]

# position_data = nutrient_data['group']
# nutrient_data = nutrient_data.iloc(axis = 1)[:22]

## 음식 data
food_dict = dict(nutrient_data['name'])

## 식단 data
morning = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data/real_menu_morning.csv', encoding='CP949')
morning = morning.iloc(axis = 1)[1:]
lunch = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data/real_menu_lunch.csv', encoding='CP949')
lunch = lunch.iloc(axis = 1)[1:]
afternoon = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data/real_menu_afternoon.csv', encoding='CP949')
afternoon = afternoon.iloc(axis = 1)[1:]
dinner = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data/real_menu_dinner.csv', encoding='CP949')
dinner = dinner.iloc(axis = 1)[1:]

diet_data = pd.concat([morning, lunch, afternoon, dinner], axis = 1)
diet_data.fillna("empty", inplace = True)
diet_data.insert(loc = 0, column = "Start", value = ["시작"] * diet_data.shape[0])
diet_data.insert(loc = diet_data.shape[1], column = "End", value = ["종료"] * diet_data.shape[0])

# Mapping from food to token
diet_data_np = food_to_token(diet_data, nutrient_data)

# Affinity Propagaion 클러스터링 
food_ap_label, ap_cluster_table = Affinity_Propagation(nutrient_data)

# Mapping from token to cluster id
cluster_data_np = token_to_cluster(diet_data_np, food_ap_label)

## 배치 데이터로 만들기 
BATCH_SIZE = 32
tf_dataset = tf.data.Dataset.from_tensor_slices(diet_data_np)
tf_dataset = tf_dataset.batch(BATCH_SIZE, drop_remainder = True)

# %%
# IDF 계산하기 위한 행렬 Action - State 만들기
action_state_mat = np.zeros([len(food_dict), diet_data_np.shape[1]])
for i in range(diet_data_np.shape[0]):
    for j in range(diet_data_np.shape[1]):
        action = diet_data_np[i, j]
        action_state_mat[int(action), j] += 1

# IDF 계산
action_state_bin = action_state_mat
action_state_bin[action_state_bin > 0] = 1
state_freq = np.sum(action_state_bin, axis = 1) # 각 action이 등장한 state의 빈도
inverse_state_freq = tf.math.log(16 / (1 + state_freq[5]))