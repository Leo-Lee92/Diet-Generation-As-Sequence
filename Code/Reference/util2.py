#%%
import numpy as np
import tensorflow as tf
import os
import glob

"""
Environment 클래스와 Agent 클래스에 포함시킬 수 없거나, 포함시킬 필요가 없는 함수들
"""

# 시퀀스를 문장으로 변환해주는 함수
def sequence_to_sentence(sequence_list, food_dict):
    """
    시퀀스를 문장으로 변환해주는 함수
    """
    gen_food_list = []

    for i, val in enumerate(sequence_list):
        each_food = []

        for j, val2 in enumerate(val):
            each_food.append(food_dict[val2])

        gen_food_list.append(each_food)

    return(gen_food_list)

# 가능성 있는 행위들 중 특정 행위를 선택하는 함수
def get_action(preds, option):
    """
    가능성 있는 행위들 중 특정 행위를 선택하는 함수. (신경망의 마지막 softmax층이 반환하는 확률벡터 기반의 확률적 샘플링)
    """
    # stochastic policy
    if option == "prob":
        probas = np.random.choice(preds, size = 1, p = preds)
        action = np.where(preds == probas)[0][0] # 선택된 액션

    # greedy policy
    elif option == "max":
        action = np.argmax(preds)
    
    # random policy
    elif option == "random":
        probas = np.random.choice(preds, size = 1)

        if len(np.where(preds == probas)[0]) >= 2:
            action = np.random.choice(np.where(preds == probas)[0], size = 1)[0]
        else:
            action = np.where(preds == probas)[0][0] # 선택된 액션

    action_prob = preds[action] # 선택된 액션의 확률
    return action, action_prob

def get_reward_ver2(nutrient_state, done):
    '''
    score_vector -> reward 뽑는 함수
    행위로부터 결정된 보상을 제공하는 함수
    '''

    '''
    영양보상 계산
    '''
    nutrient_reward = 0
    nutrient_reward_set = np.zeros([15])

    total_calorie = nutrient_state[1 - 1] # 'Energy'
    total_c = nutrient_state[2 - 1] * 4 # 'Carbohydrate' (kcal)
    total_p = nutrient_state[6 - 1] * 4 # 'Protein' (kcal)
    total_f = nutrient_state[3 - 1] * 9 # 'Fat' (kcal)
    total_satufat = nutrient_state[4 - 1] * 9 # 'Saturated Fatty acid' (kcal)
    total_transfat = nutrient_state[5 - 1] * 9 # 'Trans fatty acid' (kcal)
    total_protein_gram = nutrient_state[6 - 1] # 'Protein'
    total_dietary_gram = nutrient_state[7 - 1] # 'Total Dietary Fiber'
    total_calcium_gram = nutrient_state[8 - 1] # 'Calcium'
    total_iron_gram = nutrient_state[9 - 1] # 'Iron'
    total_sodium_gram = nutrient_state[10 - 1] # 'Sodium'
    total_phosphorus_gram = nutrient_state[11 - 1] # 'Phosphorus'
    total_vitaA_gram = nutrient_state[12 - 1] # 'Vitamin A' 
    total_vitaC_gram = nutrient_state[17 - 1] # 'Total Ascorbic Acid'
    total_vitaD_gram = nutrient_state[18 - 1] # 'Vitamin D (Ergocalciferol + Cholecalciferol)'

    # 영양보상 1. 총 칼로리 (kcal)
    if total_calorie >= 1008 and total_calorie <= 1232:
        nutrient_reward += 1
        nutrient_reward_set[0] +=1

    # 영양보상 2. 탄수화물 (kcal)
    if total_c >= 554.4 and total_c <= 862.4:
        nutrient_reward += 1
        nutrient_reward_set[1] +=1

    # 영양보상 3. 단백질 (kcal)
    if total_p >= 70.56 and total_p <= 246.4:
        nutrient_reward += 1
        nutrient_reward_set[2] +=1

    # 영양보상 4. 지방 (kcal)
    if total_f >= 151.2 and total_f <= 369.6:
        nutrient_reward += 1
        nutrient_reward_set[3] +=1

    # 영양보상 5. 포화지방 (kcal)
    if total_satufat >= 0 and total_satufat <= 29.568:
        nutrient_reward += 1
        nutrient_reward_set[4] +=1

    # 영양보상 6. 트랜스지방 (kcal)
    if total_transfat >= 0 and total_transfat <= 3.696:
        nutrient_reward += 1
        nutrient_reward_set[5] +=1

    # 영양보상 7. 지방 (gram)
    if total_protein_gram >= 18 and total_protein_gram <= 22:
        nutrient_reward += 1
        nutrient_reward_set[6] +=1

    # 영양보상 8. 식이섬유 
    if total_dietary_gram >= 10 and total_dietary_gram <= 30:
        nutrient_reward += 1
        nutrient_reward_set[7] +=1

    # 영양보상 9. 칼슘 (gram)
    if total_calcium_gram >= 400 and total_calcium_gram <= 2500:
        nutrient_reward += 1
        nutrient_reward_set[8] +=1

    # 영양보상 10. 철 (gram)
    if total_iron_gram >= 6 and total_iron_gram <= 40:
        nutrient_reward += 1
        nutrient_reward_set[9] +=1

    # 영양보상 11. 나트륨 (gram)
    if total_sodium_gram >= 500 and total_sodium_gram <= 1000:
        nutrient_reward += 1
        nutrient_reward_set[10] +=1

    # 영양보상 12. 인 (gram)
    if total_phosphorus_gram >= 300 and total_phosphorus_gram <= 3000:
        nutrient_reward += 1
        nutrient_reward_set[11] +=1

    # 영양보상 13. 비타민 A (gram)
    if total_vitaA_gram >= 230 and total_vitaA_gram <= 700:
        nutrient_reward += 1
        nutrient_reward_set[12] +=1

    # 영양보상 14. 비타민 C (gram)
    if total_vitaC_gram >= 30 and total_vitaC_gram <= 500:
        nutrient_reward += 1
        nutrient_reward_set[13] +=1

    # 영양보상 15. 비타민 D (gram)   
    if total_vitaD_gram >= 2 and total_vitaD_gram <= 35:
        nutrient_reward += 1
        nutrient_reward_set[14] +=1

    # 만약 모든 영양보상 달성시 done = 1 -> 시퀀스 생성을 중간에 종료할 수 있음.
    if nutrient_reward == 14:
        done += 1
        # nutrient_reward_set[14] +=1

    # return nutrient_reward, done, nutrient_reward_set, composition_reward
    return nutrient_reward, done, nutrient_reward_set

## 주어진 식단의 영양소 계산하기
def get_score_vector(diet, nutrient_data):
    '''
    diet는 array 타입의 tokenize된 식단 벡터이어야 함
    '''
    # 영양점수
    target_foods_mat = nutrient_data.iloc[diet, 1:]
    nutrient_vector = target_foods_mat.sum(axis = 0)

    return nutrient_vector

def reward_calculator(score_vector, pre_score_vector):
    diff = np.array(score_vector) - np.array(pre_score_vector)
    reward = diff + 1 # If add +1, then the composition of samples that add zero nutrition score is considered.
    return tf.convert_to_tensor(reward, dtype = tf.float32), diff

'''
reward calculation function for RL Tuner
'''
def reward_calculator_v2(score_vector, log_reward, pre_score_vector):
    diff = np.array(score_vector) - np.array(pre_score_vector)

    # calculate total_reward by adding nutrient_reward (diff) and composition reward (log_reward)
    reward = diff + log_reward
    # reward = np.exp(diff) + log_reward

    return tf.convert_to_tensor(reward, dtype = tf.float32), diff


def advantage_truncate(advantage):
    # 일종의 마스킹 !
    advantage = np.array(advantage)
    advantage[advantage > 1] += 1
    advantage[advantage == 0] = 1
    advantage[advantage < 0] = 0
    return tf.convert_to_tensor(advantage, dtype = tf.float32)

'''
데이터 요약 함수 ver.1
플로팅을 위해 타임스텝 (epoch), 배치당 평균 보상 또는 진짜일 확률 (value)으로 구성된 이벤트를 쌓는 함수
이벤트 = np.array([epoch, value])
'''
def summarize_episode(epoch, value, agent, network, condition, stacked_data):
    if agent == "T":
        event = np.append(epoch, value.numpy())
        event = np.append(event, agent).reshape((1, 3))
        event = np.append(event, network).reshape((1, 4))
        event = np.append(event, condition).reshape((1, 5))

        stacked_data = np.append(stacked_data, event, axis = 0)

    else:
        event = np.append(epoch, value.numpy())
        event = np.append(event, agent).reshape((1, 3))
        event = np.append(event, network).reshape((1, 4))
        event = np.append(event, condition).reshape((1, 5))

        stacked_data = np.append(stacked_data, event, axis = 0)

    return stacked_data

'''
Calculate new parameter for Target_Actor
'''
def calculate_new_weights(TA_weights, A_weights, eta):

    # Update Target Actor gradually
    new_weight_list = []
    for i in range(len(TA_weights)):
        a = tf.math.multiply((1 - eta), TA_weights[i])
        b = tf.math.multiply(eta, A_weights[i])
        new_weight_list.append(a + b)

    return new_weight_list

def merge_result_files(path, measure, network):
    if measure == "Mean Reward":
        if network == "MLP":
            all_files = sorted(glob.glob(path + "/*MLP-measure1.csv"), key = os.path.getmtime)
            li = []
            for filename in all_files:
                df = pd.read_csv(filename, engine = 'python', header = 0)
                li.append(df)
                frame = pd.concat(li, axis = 0, ignore_index = True)
        else:
            all_files = sorted(glob.glob(path + "/*RNN-measure1.csv"), key = os.path.getmtime)
            li = []
            print(all_files)
            for filename in all_files:
                df = pd.read_csv(filename, engine = 'python', header = 0)
                li.append(df)
                frame = pd.concat(li, axis = 0, ignore_index = True)

    elif measure == "Prob":
        if network == "MLP":
            all_files = sorted(glob.glob(path + "/*MLP-measure2.csv"), key = os.path.getmtime)
            li = []
            for filename in all_files:
                df = pd.read_csv(filename, engine = 'python', header = 0)
                li.append(df)
                frame = pd.concat(li, axis = 0, ignore_index = True)
        else:
            all_files = sorted(glob.glob(path + "/*RNN-measure2.csv"), key = os.path.getmtime)
            li = []
            for filename in all_files:
                df = pd.read_csv(filename, engine = 'python', header = 0)
                li.append(df)
                frame = pd.concat(li, axis = 0, ignore_index = True)
    return frame, all_files

# %%
