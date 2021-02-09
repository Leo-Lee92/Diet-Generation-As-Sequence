# %%
from util import *
import tensorflow as tf
import numpy as np
import pandas as pd
import copy
import re

## 영양소 data
# raw_data = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data/foods2.csv', encoding='CP949')
raw_data = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data/new_nutrition.csv', encoding='CP949')

empty_vector = pd.DataFrame(0, columns = raw_data.columns, index = [0])
empty_vector['name'] = "empty"
empty_vector['Class'] = "빈칸"
empty_vector['dish_class1'] = "빈칸"
empty_vector['dish_class2'] = "빈칸"
empty_vector['meal_class'] = "빈칸"
raw_data = pd.concat([empty_vector, raw_data]).reset_index(drop = True)

start_vector = pd.DataFrame(0, columns = raw_data.columns, index = [0])
start_vector['name'] = "시작"
start_vector['Class'] = "앞뒤"
start_vector['dish_class1'] = "앞뒤"
start_vector['dish_class2'] = "앞뒤"
start_vector['meal_class'] = "앞뒤"
raw_data = pd.concat([raw_data, start_vector]).reset_index(drop = True)

end_vector = pd.DataFrame(0, columns = raw_data.columns, index = [0])
end_vector['name'] = "종료"
end_vector['Class'] = "앞뒤"
end_vector['dish_class1'] = "앞뒤"
end_vector['dish_class2'] = "앞뒤"
end_vector['meal_class'] = "앞뒤"
raw_data = pd.concat([raw_data, end_vector]).reset_index(drop = True)

## 메뉴 data
nutrient_feature = list(raw_data.columns.values)
nutrient_feature = [e for e in nutrient_feature if e not in ["Weight", "Class", "dish_class1", "dish_class2", "meal_class"]]
nutrient_data = raw_data.loc(axis = 1)[nutrient_feature]
nutrient_data['name'] = nutrient_data['name'].str.replace(pat=r'[^\w]', repl=r'', regex=True)

## 메뉴 dictionary
food_dict = dict(raw_data['name'])

## 메뉴별 category data
nutrient_feature = list(raw_data.columns.values)
nutrient_feature = [e for e in nutrient_feature if e in ["name", "Class", "dish_class1", "dish_class2", "meal_class"]]
category_data = raw_data.loc(axis = 1)[nutrient_feature]
category_data["dish_class2"] = category_data["dish_class2"].astype('category')

## 식단 data
# morning = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data/real_menu_morning.csv', encoding='CP949')
# morning = morning.iloc(axis = 1)[1:]
# lunch = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data/real_menu_lunch.csv', encoding='CP949')
# lunch = lunch.iloc(axis = 1)[1:]
# afternoon = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data/real_menu_afternoon.csv', encoding='CP949')
# afternoon = afternoon.iloc(axis = 1)[1:]
# dinner = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data/real_menu_dinner.csv', encoding='CP949')
# dinner = dinner.iloc(axis = 1)[1:]
# diet_data = pd.concat([morning, lunch, afternoon, dinner], axis = 1)

diet_data = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data/new_diet.csv', encoding='CP949')
diet_data = diet_data.replace('[^\w]', '', regex=True)
diet_data.fillna("empty", inplace = True)
menus_in_dietDB = set(np.unique( np.char.strip(diet_data.values.flatten().astype('str')) ))                                  # dietDB에 등장한 menu들
menus_in_nutritionDB = set(np.unique( np.char.strip(nutrient_data['name'].values.flatten().astype('str'))))                  # nutritionDB에 등장한 menu들
menus_only_in_dietDB = menus_in_dietDB.difference(menus_in_nutritionDB)                                                      # dietDB에만 존재하는 menu들
pd.DataFrame(menus_only_in_dietDB).to_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data/menus_only_in_dietDB.csv', encoding="utf-8-sig")
diet_data = diet_data.replace(menus_only_in_dietDB, "empty")                                                                 # dietDB에서 nutritionDB에 없는 menu들 empty로 바꿔주기
diet_data.insert(loc = 0, column = "Start", value = ["시작"] * diet_data.shape[0])
diet_data.insert(loc = diet_data.shape[1], column = "End", value = ["종료"] * diet_data.shape[0])

# Mapping from food to token
# diet_data_np = food_to_token(diet_data, nutrient_data)
diet_data_np = food_to_token(diet_data, nutrient_data, empty_delete = True, num_empty = 2)

# Build the menu transition matrix
p_prob_mat = transition_matrix(diet_data_np, food_dict)

# Mapping from diet_data_np to binary incidence format
incidence_data = diet_to_incidence(diet_data_np, food_dict)

## 알러지 유발 메뉴 리스트
alergy_trigger_list = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data/menu_with_meat.csv')
alergy_trigger_list.columns = ['menu', 'class']
alergy_menu_token = [np.where(np.array(list(food_dict.values())) == alergy_menu)[0][0] for alergy_menu in alergy_trigger_list['menu']]
alergy_menu_onehot = tf.one_hot(alergy_menu_token, len(food_dict))
alergy_menu_vector = np.sum(alergy_menu_onehot, axis = 0).reshape(-1, 1)


# 텐서플로 데이터셋 자료형으로 변환후 배치 슬라이싱 해주기
BATCH_SIZE = diet_data_np.shape[0]
BUFFER_SIZE = int(BATCH_SIZE * (diet_data_np.shape[0] / BATCH_SIZE))

tf_dataset = tf.data.Dataset.from_tensor_slices(diet_data_np)   # 텐서슬라이싱 해주기
tf_dataset = tf_dataset.shuffle(buffer_size = BUFFER_SIZE, seed = 1234, reshuffle_each_iteration = False).take(BUFFER_SIZE)  # BUFFER_SIZE만큼 랜덤 샘플링
tf_dataset = tf_dataset.batch(BATCH_SIZE, drop_remainder = True) # BATCH_SIZE로 배치화 해주기

tf_dataset_update = tf.data.Dataset.from_tensor_slices(diet_data_np)   # 텐서슬라이싱 해주기
tf_dataset_update = tf_dataset_update.shuffle(buffer_size = BUFFER_SIZE, seed = 1234, reshuffle_each_iteration = False).take(BUFFER_SIZE)  # BUFFER_SIZE만큼 랜덤 샘플링
tf_dataset_update = tf_dataset_update.batch(BATCH_SIZE, drop_remainder = True)
# tf_dataset_update = tf_dataset

# truncated_cluster_diet_np = np.array(list(tf_dataset))[:, 1, :, :]
# truncated_cluster_diet_np = np.reshape(truncated_cluster_diet_np, (-1, 16))
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