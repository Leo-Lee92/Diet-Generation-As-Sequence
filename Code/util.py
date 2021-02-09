#%%
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import glob
from tqdm import tqdm
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import copy

"""
Environment 클래스와 Agent 클래스에 포함시킬 수 없거나, 포함시킬 필요가 없는 함수들
"""
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def do_curriculum(do, epoch, num_epochs, len_seq, curriculum_step):

    if do == True:
        '''
        Curriculum Learning Setting
        '''
        # 만약 현 epoch가 num_epoch의 60%를 초과하여 진행된 상태라면 (점진적으로 XE-loss에서 RL-loss로 학습)
        if epoch + 1 >= 0.5 * num_epochs:                           # 전체 epoch의 60%에 해당하는 epoch까지 (= XENT epoch까지)
            N = 30                                                  # 매 N epoch마다 (= XE+R epoch 마다)
            if (epoch + 1) % N == 0:     
                curriculum_range = copy.deepcopy(len_seq)           # curriculum_range를 전체 구간으로 초기화하라
                curriculum_step += 1                                # curriculum learning 해야할 step 누적
                curriculum_range -= curriculum_step                 # curriculum_range가 누적 step만큼 줄어듦

                if curriculum_range < 1:                            # 만약 curriculum_range가 1보다 작아진다면
                    curriculum_range = copy.deepcopy(len_seq)       # curriculum_range를 전체 구간으로 초기화하라
                    curriculum_step = 0                             # curriculum_step을 0으로 초기화 하라
            else:
                curriculum_range = copy.deepcopy(len_seq) - curriculum_step

        # 만약 현 epoch이 전체 epochs의 80% 이하진행된 상태라면 전 영역 커리큘럼 학습 (전영역 XE-loss로 학습)
        else:
            curriculum_range = "All"

    else:
        curriculum_range = None

    return curriculum_range, curriculum_step

def meal_hit_score(diet, category_data):
    '''
    최소한 달성해야 하는 식단의 meal-level 구성 관점에서 매긴 점수
    > Position 기반 Composition Measure
    만점 9점.
    '''
    score = 0
    score_vector = np.empty([0, 9])

    # 만약 "빈칸" meal_class가 index 0, 1, 8, 15를 제외하고 나왔다면
    true_idx = set([2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14])
    diet_idx = set(np.where(category_data.iloc(axis = 0)[diet]['meal_class'] == "빈칸")[0])
    score += int(diet_idx.issubset(true_idx))
    
    if int(diet_idx.issubset(true_idx)) > 0:
        score_vector = np.append(score_vector, 1)
    else:
        score_vector = np.append(score_vector, 0)

    # 만약 "앞뒤" meal_class가 index 0, 15에 있다면
    true_idx = set([0, 15])
    diet_idx = set(np.where(category_data.iloc(axis = 0)[diet]['meal_class'] == "앞뒤")[0])
    score += int(diet_idx.issubset(true_idx))

    if int(diet_idx.issubset(true_idx)) > 0:
        score_vector = np.append(score_vector, 1)
    else:
        score_vector = np.append(score_vector, 0)

    # 만약 "간식" meal_class가 index 1, 2, 8, 9에 있다면
    true_idx = set([1, 2, 8, 9])
    diet_idx = set(np.where(category_data.iloc(axis = 0)[diet]['meal_class'] == "간식")[0])
    score += int(diet_idx.issubset(true_idx))

    if int(diet_idx.issubset(true_idx)) > 0:
        score_vector = np.append(score_vector, 1)
    else:
        score_vector = np.append(score_vector, 0)

    # 만약 "간식저녁" meal_class가 index 1, 2, 8, 9, 10, 11, 12, 13, 14에 있다면
    true_idx = set([1, 2, 8, 9, 10, 11, 12, 13, 14])
    diet_idx = set(np.where(category_data.iloc(axis = 0)[diet]['meal_class'] == "간식저녁")[0])
    score += int(diet_idx.issubset(true_idx))

    if int(diet_idx.issubset(true_idx)) > 0:
        score_vector = np.append(score_vector, 1)
    else:
        score_vector = np.append(score_vector, 0)

    # 만약 "간식점심" meal_class가 index 1, 2, 8, 9, 3, 4, 5, 6, 7에 있다면
    true_idx = set([1, 2, 8, 9, 3, 4, 5, 6, 7])
    diet_idx = set(np.where(category_data.iloc(axis = 0)[diet]['meal_class'] == "간식점심")[0])
    score += int(diet_idx.issubset(true_idx))

    if int(diet_idx.issubset(true_idx)) > 0:
        score_vector = np.append(score_vector, 1)
    else:
        score_vector = np.append(score_vector, 0)

    # 만약 "간식점심저녁" meal_class가 index 1, 2, 8, 9, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14에 있다면
    true_idx = set([1, 2, 8, 9, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14])
    diet_idx = set(np.where(category_data.iloc(axis = 0)[diet]['meal_class'] == "간식점심저녁")[0])
    score += int(diet_idx.issubset(true_idx))

    if int(diet_idx.issubset(true_idx)) > 0:
        score_vector = np.append(score_vector, 1)
    else:
        score_vector = np.append(score_vector, 0)

    # 만약 "저녁" meal_class가 index 10, 11, 12, 13, 14에 있다면
    true_idx = set([10, 11, 12, 13, 14])
    diet_idx = set(np.where(category_data.iloc(axis = 0)[diet]['meal_class'] == "저녁")[0])
    score += int(diet_idx.issubset(true_idx))

    if int(diet_idx.issubset(true_idx)) > 0:
        score_vector = np.append(score_vector, 1)
    else:
        score_vector = np.append(score_vector, 0)

    # 만약 "점심" meal_class가 index 3, 4, 5, 6, 7에 있다면
    true_idx = set([3, 4, 5, 6, 7])
    diet_idx = set(np.where(category_data.iloc(axis = 0)[diet]['meal_class'] == "점심")[0])
    score += int(diet_idx.issubset(true_idx))

    if int(diet_idx.issubset(true_idx)) > 0:
        score_vector = np.append(score_vector, 1)
    else:
        score_vector = np.append(score_vector, 0)

    # 만약 "점심저녁" meal_class가 index 3, 4, 5, 6, 7, 10, 11, 12, 13, 14에 있다면
    true_idx = set([3, 4, 5, 6, 7, 10, 11, 12, 13, 14])
    diet_idx = set(np.where(category_data.iloc(axis = 0)[diet]['meal_class'] == "점심저녁")[0])
    score += int(diet_idx.issubset(true_idx))

    if int(diet_idx.issubset(true_idx)) > 0:
        score_vector = np.append(score_vector, 1)
    else:
        score_vector = np.append(score_vector, 0)

    return score/len(score_vector), score_vector

def dish_hit_score(diet, category_data):
    '''
    최소한 달성해야 하는 식단의 dish-level 구성 관점에서 매긴 점수
    > Position 기반 Composition Measure
    1. 0번, 15번 index에는 "앞뒤" 서브클래스2
    2. 1,2, 8,9번 index에는 "간식" 서브클래스2
    3. 3,10번 index에는 "밥" 서브클래스2
    4. 4,11번 index에는 "국" 서브클래스2
    5. 5,6,12,13번 index에는 "반찬" 서브클래스2
    6. 7,14번 index에는 "김치" 서브클래스2
    7. 3,4,10,11번 index에는 "밥국" 서브클래스2
    8. morning snack1 & afternoon snack1 빼고 "빈칸" 서브클래스2는 전부 나올 수 있음

    만점 8점.
    '''
    score = 0
    score_vector = np.empty([0, 8])

    # 만약 "빈칸" 서브클래스2가 index 0, 1, 8, 15를 제외하고 나왔다면
    true_idx = set([2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14])
    diet_idx = set(np.where(category_data.iloc(axis = 0)[diet]['dish_class2'] == "빈칸")[0])
    score += int(diet_idx.issubset(true_idx))
    
    if int(diet_idx.issubset(true_idx)) > 0:
        score_vector = np.append(score_vector, 1)
    else:
        score_vector = np.append(score_vector, 0)

    # 만약 "앞뒤" 서브클래스2가 index 0, 15에 있다면
    true_idx = set([0, 15])
    diet_idx = set(np.where(category_data.iloc(axis = 0)[diet]['dish_class2'] == "앞뒤")[0])
    score += int(diet_idx.issubset(true_idx))

    if int(diet_idx.issubset(true_idx)) > 0:
        score_vector = np.append(score_vector, 1)
    else:
        score_vector = np.append(score_vector, 0)

    # 만약 "간식" 서브클래스2가 index 1,2,8,9에 있다면
    true_idx = set([1, 2, 8, 9])
    diet_idx = set(np.where(category_data.iloc(axis = 0)[diet]['dish_class2'] == "간식")[0])
    score += int(diet_idx.issubset(true_idx))

    if int(diet_idx.issubset(true_idx)) > 0:
        score_vector = np.append(score_vector, 1)
    else:
        score_vector = np.append(score_vector, 0)

    # 만약 "밥" 서브클래스2가 index 3,10에 있다면
    true_idx = set([3, 10])
    diet_idx = set(np.where(category_data.iloc(axis = 0)[diet]['dish_class2'] == "밥")[0])
    score += int(diet_idx.issubset(true_idx))

    if int(diet_idx.issubset(true_idx)) > 0:
        score_vector = np.append(score_vector, 1)
    else:
        score_vector = np.append(score_vector, 0)

    # 만약 "국" 서브클래스2가 index 4,11에 있다면
    true_idx = set([4, 11])
    diet_idx = set(np.where(category_data.iloc(axis = 0)[diet]['dish_class2'] == "국")[0])
    score += int(diet_idx.issubset(true_idx))

    if int(diet_idx.issubset(true_idx)) > 0:
        score_vector = np.append(score_vector, 1)
    else:
        score_vector = np.append(score_vector, 0)

    # 만약 "반찬" 서브클래스2가 index 5,6,12,13에 있다면
    true_idx = set([5, 6, 12, 13])
    diet_idx = set(np.where(category_data.iloc(axis = 0)[diet]['dish_class2'] == "반찬")[0])
    score += int(diet_idx.issubset(true_idx))

    if int(diet_idx.issubset(true_idx)) > 0:
        score_vector = np.append(score_vector, 1)
    else:
        score_vector = np.append(score_vector, 0)

    # 만약 "김치" 서브클래스가 index 7, 14에 있다면
    true_idx = set([7, 14])
    diet_idx = set(np.where(category_data.iloc(axis = 0)[diet]['dish_class2'] == "김치")[0])
    score += int(diet_idx.issubset(true_idx))

    if int(diet_idx.issubset(true_idx)) > 0:
        score_vector = np.append(score_vector, 1)
    else:
        score_vector = np.append(score_vector, 0)

    # 만약 "밥국" 서브클래스가 index 3,4,10,11에 있다면
    true_idx = set([3, 4, 10, 11])
    diet_idx = set(np.where(category_data.iloc(axis = 0)[diet]['dish_class2'] == "밥국")[0])
    score += int(diet_idx.issubset(true_idx))

    if int(diet_idx.issubset(true_idx)) > 0:
        score_vector = np.append(score_vector, 1)
    else:
        score_vector = np.append(score_vector, 0)

    return score/len(score_vector), score_vector

def plot_reward(dir_reward_df):
    reward_df = pd.read_csv(dir_reward_df)

    # per_epoch_len = reward_df[reward_df['epoch'] == 0].shape[0]

    # all_idx = np.arange(reward_df.shape[0])
    # all_idx_partition = all_idx // (len(all_idx) / 10)
    # all_first_idx_of_partition = [np.where(all_idx_partition == i)[0][0] for i in np.unique(all_idx_partition)]
    # tick_marker = [str(v + 1) + 'k' for v, k in enumerate(all_first_idx_of_partition)]
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    sns.set_palette(sns.color_palette(colors))

    if 'method' in reward_df.columns:
        # plt.xlabel('epoch')
        sns.set(style = 'darkgrid')
        g = sns.lineplot(data = reward_df, x = 'epoch', y = 'reward', hue = 'method', ci = 'sd')
        # g.set_xticks(all_first_idx_of_partition)
        # g.set_xticklabels(tick_marker)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='lower right', borderaxespad=0.)
        plt.legend(loc='lower right', title = 'method')

        figure_dir = '/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/figures/training_reward.png'
        plt.axhline(y = 9.29, color = "#d62728", linestyle = "-")
        plt.savefig(figure_dir, bbox_inches='tight')
    else:
        sns.lineplot(data = reward_df, x = 'epoch', y = 'reward', ci = 'sd')

# def make_nutrient_df(diet, total_nutrient_df):
#     # 실제 식단의 nutrient 벡터 만들기
#     nutrient_vector = get_score_vector(diet, nutrient_data)[:20]
#     marginal_nutrient_df = pd.DataFrame(nutrient_vector).transpose()
#     total_nutrient_df = total_nutrient_df.append(marginal_nutrient_df)        
#     return total_nutrient_df

def make_matrix_for_tsne(total_nutrient_df_gen):
    global total_nutrient_df, method_label
    total_nutrient_df = pd.concat([pd.DataFrame(total_nutrient_df), pd.DataFrame(total_nutrient_df_gen)])
    method_label = np.append(method_label, np.repeat(cp_dir.split('_')[len(cp_dir.split('_')) - 2], total_nutrient_df_gen.shape[0]))

    return total_nutrient_df, method_label

def tsne_mapping_multiple_gen(total_nutrient_df_real, total_nutrient_df):
    total_nutrient_df = total_nutrient_df_real.append(total_nutrient_df)

    tsne_model = TSNE(learning_rate = 100, random_state = 1234)
    tsne_values = tsne_model.fit_transform(total_nutrient_df)
    return tsne_values

def tsne_mapping(total_nutrient_df_real, total_nutrient_df_gen):
    total_nutrient_df = total_nutrient_df_real.append(total_nutrient_df_gen)

    label_real = pd.Series(np.zeros((1, total_nutrient_df_real.shape[0]))[0], name = "label")
    label_gen = pd.Series(np.ones((1, total_nutrient_df_gen.shape[0]))[0], name = "label")
    label_total = pd.concat([label_real, label_gen], ignore_index = True)

    tsne_model = TSNE(learning_rate = 100, random_state = 1234)
    tsne_values = tsne_model.fit_transform(total_nutrient_df)
    tsne_values = np.append(tsne_values, np.array(label_total).reshape(-1, 1), axis = 1)
    return tsne_values

def tsne_plot(tsne_values, method_label):

    method_label = np.array(method_label)
    method_label = np.append(np.repeat('real', 1072), method_label)

    tsne_df = pd.DataFrame(tsne_values)
    tsne_df['method'] = method_label
    tsne_df.columns = ["x", "y", "method"]

    colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]
    sns.set_palette(sns.color_palette(colors))
    sns.scatterplot(data = tsne_df, x = "x", y = "y", hue = "method", style = "method", s = 20, alpha = .5)
    figure_dir = '/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/figures/tsne_mapping_plot.png'
    plt.savefig(figure_dir, bbox_inches='tight')

def rewards_matrix(epoch, rewards):

    rewards = np.array(rewards).reshape(-1, 1)
    # quantile_rewards = np.array([np.quantile(rewards, q) for q in [0, .25, .5, .75, 1.]])

    epochs = np.repeat(epoch, len(rewards))
    epochs = epochs.reshape(-1, 1)

    samples = np.array(range(len(rewards)))
    samples = samples.reshape(-1, 1)

    # mean_rewards = np.repeat(np.mean(rewards), len(quantile_rewards))
    # mean_rewards = mean_rewards.reshape(-1, 1)

    # rewards = quantile_rewards.reshape(-1, 1)
    per_epoch_rewards = pd.DataFrame(np.concatenate((epochs, rewards, samples), axis = 1))
    
    # mean_rewards = np.append(per_epoch_mean_rewards, mean_rewards)
    return per_epoch_rewards

def save_reward_df(reward_df, model, eb, bs, lr, num_epochs):
    dir_file_name = '/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/results/' + model + '_eb=' + str(eb) + '_bs=' + str(bs) + '_lr=' + str(lr) + '_epoch=' + str(num_epochs) + '_rewards.csv'
    if not os.path.exists(dir_file_name):
        reward_df.to_csv(dir_file_name, index=False, mode='w')
    else:
        reward_df.to_csv(dir_file_name, index=False, mode='a', header=False)
    
    return dir_file_name

def transition_matrix(diet_data_np, food_dict):
    p_prob_mat = np.full([len(food_dict), len(food_dict)], fill_value = 0, dtype = np.float64)
    for diet_idx in range(np.shape(diet_data_np)[0]):
        for (i, j) in zip(diet_data_np[diet_idx, :].astype(int), diet_data_np[diet_idx, :][1:].astype(int)):
            # print('i : {}, j : {}'.format(i, j))
            p_prob_mat[i, j] += 1

    p_prob_mat = p_prob_mat / p_prob_mat.sum(axis = 1, keepdims = True) 
    p_prob_mat = np.nan_to_num(p_prob_mat, nan = 0.0, copy = True) + (1/len(food_dict))

    return p_prob_mat

def RollOutSamplar(incidence_data, t, inputs):
    rollout_policy = incidence_data[:, t] / sum(incidence_data[:, t])
    rollout_policy = tf.reshape(rollout_policy, shape = (1, -1))
    rollout_policy_tensor = tf.repeat(rollout_policy, repeats = inputs.shape[0], axis = 0)
    return rollout_policy_tensor

def load_weights_from_checkpoint(loaded_checkpoint, key_list, network = None):
    '''
    network : checkpoint에 저장된 네트워크 인스턴스의 이름
    '''
    weight_list = []
    for key in key_list:
        key2 = key.split('/')
        if key2[0] == network:
            weight_list.append(loaded_checkpoint.get_tensor(key))
    
    return weight_list

# Mapping diet data to incidence data
def diet_to_incidence(diet_data_np, food_dict):
    incidence_mat = tf.zeros([len(food_dict), diet_data_np.shape[1]])

    for i in range(diet_data_np.shape[0]):
        incidence_mat += tf.transpose(tf.one_hot(diet_data_np[i, :], depth = len(food_dict)))

    return incidence_mat
    
# Mapping food to token
def food_to_token(diet_data, nutrient_data, empty_delete = False, num_empty = 2):
    '''
    empty_delete : empty 갯수 지정해서 지정한 갯수 이상 포함된 식단ㅇ들 제거할지 여부. default = False이고, 이 경우 empty로 꽉찬 행만 제거
    num_empty : empty 갯수
    '''

    diet_data_np = np.zeros([diet_data.shape[0], diet_data.shape[1]])
    # Mapping from food to token
    ## 영양소 data기준으로 식단 data tokenization
    delete_list = np.array([])
    for i in tqdm(range(diet_data.shape[0])):

        empty_list = np.array([])

        for j in range(diet_data.shape[1]):
            try:
                # tokenization
                diet_data_np[i, j] = nutrient_data[nutrient_data['name'] == diet_data.iloc[i, j]].index[0]

                # 각 식단마다 등장한 empty의 갯수를 담은 리스트 생성
                if diet_data_np[i, j] == 0:
                    empty_list = np.append(empty_list, j)
                    # print('i :{}, j : {}, empty_list : {}'.format(i, j, empty_list))
            except:
                pass

        # print('diet : {}'.format(sequence_to_sentence([diet_data_np[i, :]], food_dict)))

        # i번쨰 식단에 등장한 empty의 갯수가 num_empty보다 클 경우 해당 식단 i를 삭제해야할 대상인 delete_list에 담기
        if len(empty_list) > num_empty:
            delete_list = np.append(delete_list, i)

    # print('delete_list :', delete_list)

    # 아무식단도 제공되지 않은날
    if empty_delete == False:
        non_value_idx = np.where(np.sum(diet_data_np, axis = 1) == 3453)[0] 
    else:
        non_value_idx = copy.deepcopy(delete_list)     # 점심, 저녁 중 empty가 2번 초과 등장한 경우
                                                       # 식단에 empty가 n번 이상 등장한 경우

    # 아무식단도 제공되지 않은날을 의미하는 row 제거
    diet_data_np = np.delete(diet_data_np, non_value_idx.astype(int), axis = 0) 

    return diet_data_np

# Mapping token to clsuter id
def token_to_cluster(diet_data_np, food_ap_label):
    cluster_data_np = np.empty([0, diet_data_np.shape[1]])
    for i in range(diet_data_np.shape[0]):
        cluster_data_np = np.vstack([cluster_data_np, food_ap_label[diet_data_np[i].astype('int')]])

    return cluster_data_np

# Prceeds Affinity Propagaion.
def Affinity_Propagation(nutrient_data):
    X = np.array(nutrient_data.iloc[:, 1:])     # 데이터 
    clustering = AffinityPropagation().fit(X)   # 데이터 Affinity Propagation 피팅
    food_ap_label = clustering.labels_          # 각 데이터 포인트들의 클러스터 벡터 확인
    # clustering.predict([[0, 0], [4, 4]])        # 주어진 데이터가 앞서 Fit 시킨 결과 기준으로 어느 클러스터에 속하는지 예측
    clustering.cluster_centers_                 # 학습한 데이터에서 센터 확인하기

    food_vector = np.array(range(len(np.unique(food_ap_label))))
    ap_label_freq = np.bincount(food_ap_label)
    ap_cluster_table = np.append(food_vector, ap_label_freq).reshape(2, len(ap_label_freq)).T
    return food_ap_label, ap_cluster_table


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


def get_composition_score(diet, category_data):

    meal_hit_reward = meal_hit_score(diet, category_data)[0]
    dish_hit_reward = dish_hit_score(diet, category_data)[0]
    return meal_hit_reward * dish_hit_reward

def get_reward_ver2(nutrient_state, done):
    '''
    score_vector -> reward 뽑는 함수
    행위로부터 결정된 보상을 제공하는 함수
    '''

    '''
    영양보상 계산
    '''
    nutrient_reward = 0
    nutrient_reward_set = np.zeros([13])

    total_calorie = nutrient_state[0] # 'Energy'
    total_c = nutrient_state[3] * 4 # 'Carbohydrate' (kcal)
    total_p = nutrient_state[1] * 4 # 'Protein' (kcal)
    total_p_gram = nutrient_state[1] # 'Protein' (g)
    total_f = nutrient_state[2] * 9 # 'Fat' (kcal)
    total_dietary = nutrient_state[4] # 'Total Dietary Fiber' (g)
    total_calcium = nutrient_state[5] # 'Calcium' (mg)
    total_iron = nutrient_state[6] # 'Iron' (mg)
    total_sodium = nutrient_state[7] # 'Sodium' (mg)
    total_vitaA = nutrient_state[8] + nutrient_state[9] # 'Vitamin A' = Retinol + beta-carotene (mugRAE)
    total_vitaB1 = nutrient_state[10]   # 'Vitamin B1 (Thiamine)' (mg)
    total_vitaB2 = nutrient_state[11]   # 'Vitamin B2 (Rivoflavin)' (mg)
    total_vitaC = nutrient_state[12]    # 'Vitamin C' (mg)
    total_lino = nutrient_state[13]     # 'Linoleic Acid' (g)
    total_alpha = nutrient_state[14]    # 'Alpha-Linoleic Acid' (g)
    total_EPA_DHA = nutrient_state[15] + nutrient_state[16] # 'EPA + DHA' (mg)

    # 영양보상 1. 총 열량 (kcal)
    if total_calorie >= 945 and total_calorie <= 1155:
        nutrient_reward += 1
        nutrient_reward_set[0] +=1

    # 영양보상 2. 단백질 (g)
    if total_p_gram >= 15:
        nutrient_reward += 1
        nutrient_reward_set[1] +=1

    # 영양보상 3. 총 식이섬유 (g)
    if total_dietary >= 8.25 and total_dietary <= 15:
        nutrient_reward += 1
        nutrient_reward_set[2] +=1

    # 영양보상 4. 비타민 A (mugRAE)
    if total_vitaA >= 172.5 and total_vitaA <= 562.5:
        nutrient_reward += 1
        nutrient_reward_set[3] +=1

    # 영양보상 5. 비타민 C (mg)
    if total_vitaC >= 26.25 and total_vitaC <= 382.5:
        nutrient_reward += 1
        nutrient_reward_set[4] +=1

    # 영양보상 6. 비타민 B1 (mg)
    if total_vitaB1 >= 0.3:
        nutrient_reward += 1
        nutrient_reward_set[5] +=1

    # 영양보상 7. 비타민 B2 (mg)
    if total_vitaB2 >= 0.375:
        nutrient_reward += 1
        nutrient_reward_set[6] +=1

    # 영양보상 8. 칼슘 (mg) 
    if total_calcium >= 375 and total_calcium <= 1875:
        nutrient_reward += 1
        nutrient_reward_set[7] +=1

    # 영양보상 9. 철 (mg)
    if total_iron >= 3.75 and total_iron <= 30:
        nutrient_reward += 1
        nutrient_reward_set[8] +=1

    # 영양보상 10. 나트륨 (mg)
    if total_sodium <= 1200:
        nutrient_reward += 1
        nutrient_reward_set[9] +=1

    # 영양보상 11. 리놀레산 (g)
    if total_lino >= 3.3 and total_lino <= 6.8:
        nutrient_reward += 1
        nutrient_reward_set[10] +=1

    # 영양보상 12. 알파 리놀렌산 (g)
    if total_alpha >= 0.4 and total_alpha <= 0.9:
        nutrient_reward += 1
        nutrient_reward_set[11] +=1

    # # 영양보상 13. EPA + DHA (mg)
    # if total_EPA_DHA >= 90 and total_EPA_DHA <= 180.5:
    #     nutrient_reward += 1
    #     nutrient_reward_set[12] +=1

    # 영양보상 14. 탄단지 비율 (kcal %) - 고쳐야 됨
    if ((total_c >= total_calorie * 0.55 and total_c <= total_calorie * 0.65) and 
    (total_p >= total_calorie * 0.07 and total_p <= total_calorie * 0.2) and 
    (total_f >= total_calorie * 0.15 and total_f <= total_calorie * 0.3)):
        nutrient_reward += 1
        nutrient_reward_set[12] +=1

    # return nutrient_reward, done, nutrient_reward_set, composition_reward
    return nutrient_reward, done, nutrient_reward_set

## 주어진 식단의 영양소 계산하기
def get_score_matrix(diet_batch, nutrient_data, food_dict):
    diet_batch = tf.cast(diet_batch, dtype = tf.int64) # int 타입으로

    # onehot 벡터인데 time_step이 없는 one_hot으로
    diet_batch = tf.reduce_sum(tf.one_hot(diet_batch, depth = len(food_dict)), axis = 1)

    # 영양소 DB array 형태로
    nutrient_array = np.array(nutrient_data.iloc[:, 1:])

    # 각 식단별 영양소 계산
    nutrient_vector = np.tensordot(diet_batch, nutrient_array, axes = 1)
    return nutrient_vector

## 알러지 유발 메뉴인지 판정
'''
get_score_matrix와 마찬가지로 행렬을 넣는 것임.
'''
def is_alergy_trigger(diet_batch, alergy_menu_vector, food_dict):
    diet_batch = tf.cast(diet_batch, dtype = tf.int64) # int 타입으로

    # onehot 벡터인데 time_step이 없는 one_hot으로
    diet_batch = tf.reduce_sum(tf.one_hot(diet_batch, depth = len(food_dict)), axis = 1)
    diet_batch_np = diet_batch.numpy()
    diet_batch_np[diet_batch_np >= 1] = 1   # 발생빈도가 1이상이면 1로

    # 알러지 유발메뉴 포함 식단의 인덱스 (포함 = 1, 비포함 = 0)
    alergy_diets_idx = np.where(np.dot(diet_batch_np, alergy_menu_vector).flatten() == 1)[0]

    # aa는 알러지 유발메뉴를 포함식단 0점, 유발메뉴 비포함식단 1점주는 전체 식단 샘플갯수만큼의 길이를 각진 onehot 벡터가 됨
    aa = np.ones(len(diet_batch))
    aa[alergy_diets_idx] = 0
    alergy_trigers = copy.deepcopy(aa)

    return alergy_trigers

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
