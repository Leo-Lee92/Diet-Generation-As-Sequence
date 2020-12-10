# %%
import tensorflow as tf
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm

## 영양소 data
raw_data = pd.read_csv('/home/messy92/Leo/Project_Gosin/Data/foods2.csv', encoding='CP949')
nutrient_data = raw_data.iloc(axis = 1)[:23]

empty_vector = pd.DataFrame(0, columns = nutrient_data.columns, index = [0])
empty_vector['name'] = "empty"
empty_vector['group'] = 8
nutrient_data = pd.concat([empty_vector, nutrient_data]).reset_index(drop = True)

start_vector = pd.DataFrame(0, columns = nutrient_data.columns, index = [0])
start_vector['name'] = "시작"
start_vector['group'] = 9
nutrient_data = pd.concat([nutrient_data, start_vector]).reset_index(drop = True)

end_vector = pd.DataFrame(0, columns = nutrient_data.columns, index = [0])
end_vector['name'] = "종료"
end_vector['group'] = 9
nutrient_data = pd.concat([nutrient_data, end_vector]).reset_index(drop = True)

position_data = nutrient_data['group']
nutrient_data = nutrient_data.iloc(axis = 1)[:22]

## 음식 data
food_dict = dict(nutrient_data['name'])

## 식단 data
morning = pd.read_csv('/home/messy92/Leo/Project_Gosin/Data/real_menu_morning.csv', encoding='CP949')
morning = morning.iloc(axis = 1)[1:]
lunch = pd.read_csv('/home/messy92/Leo/Project_Gosin/Data/real_menu_lunch.csv', encoding='CP949')
lunch = lunch.iloc(axis = 1)[1:]
afternoon = pd.read_csv('/home/messy92/Leo/Project_Gosin/Data/real_menu_afternoon.csv', encoding='CP949')
afternoon = afternoon.iloc(axis = 1)[1:]
dinner = pd.read_csv('/home/messy92/Leo/Project_Gosin/Data/real_menu_dinner.csv', encoding='CP949')
dinner = dinner.iloc(axis = 1)[1:]

diet_data = pd.concat([morning, lunch, afternoon, dinner], axis = 1)
diet_data.fillna("empty", inplace = True)
diet_data.insert(loc = 0, column = "Start", value = ["시작"] * diet_data.shape[0])
diet_data.insert(loc = diet_data.shape[1], column = "End", value = ["종료"] * diet_data.shape[0])

diet_data_np = np.zeros([diet_data.shape[0], diet_data.shape[1]])

## 영양소 data기준으로 식단 data tokenization
delete_list = np.array([])
for i in tqdm(range(diet_data.shape[0])):
    empty_list = np.array([])

    for j in range(diet_data.shape[1]):
        try:
            # tokenization
            diet_data_np[i, j] = nutrient_data[nutrient_data['name'] == diet_data.iloc[i, j]].index[0]

            # 각 i번째 식단마다 2, 3, 4, ..., 10, 11, 12, 13 슬롯에 등장한 empty의 갯수를 담은 리스트 생성
            if j in [2, 3, 4, 5, 6, 9, 10, 11, 12, 13] and diet_data_np[i, j] == 0:
                empty_list = np.append(empty_list, j)
        except:
            pass

    # i번쨰 식단에 등장한 empty의 갯수가 n보다 클 경우 해당 식단 i를 삭제해야할 대상인 delete_list에 담기
    if len(empty_list) > 2:
        delete_list = np.append(delete_list, i)

    # if len(np.where(diet_data_np[i, :] == 0)[0]) > 0:
    #     delete_list = np.append(delete_list, i)

non_value_idx = np.where(np.sum(diet_data_np, axis = 1) == 3453)[0] # 아무급식도 제공되지 않은날
# non_value_idx = delete_list # 점심, 저녁 중 empty가 2번 초과 등장한 경우
#                           # 식단에 empty가 n번 이상 등장한 경우

diet_data_np = np.delete(diet_data_np, non_value_idx, axis = 0) # 아무급식도 제공되지 않은날을 의미하는 row 제거


## 배치 데이터로 만들기 
BATCH_SIZE = 32
tf_dataset = tf.data.Dataset.from_tensor_slices(diet_data_np)
tf_dataset = tf_dataset.batch(BATCH_SIZE, drop_remainder = True)


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
# %%
