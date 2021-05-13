# %%
## reward 구하는 matrix 연산자 구현하기
start = time.time()  # 시작 시간 저장
for i in range(len(list(tf_dataset))):

    tmp_diet = list(tf_dataset)[i] # 식단 배치
    tmp_diet = tf.cast(tmp_diet, dtype = tf.int64) # int 타입으로

    # onehot 벡터인데 time_step이 없는 one_hot으로
    tmp_diet = tf.reduce_sum(tf.one_hot(tmp_diet, depth = len(food_dict)), axis = 1)

    # 영양소 DB array 형태로
    nutrient_array = np.array(nutrient_data.iloc[:, 1:])

    # 각 식단별 영양소 계산
    nutrient_score = np.tensordot(tmp_diet, nutrient_array , axes = 1)
    # print(nutrient_score)
print("time :", time.time() - start)

start = time.time()  # 시작 시간 저장
for i in range(len(list(tf_dataset))):
    tmp_diet = list(tf_dataset)[i] # 식단 배치
    tmp_diet = tf.cast(tmp_diet, dtype = tf.int64) # int 타입으로

    nutrient_score = np.apply_along_axis(get_score_vector, axis = 1, arr = tmp_diet, nutrient_data = nutrient_data)
    # print(nutrient_score)
print("time :", time.time() - start)


# %%
## 알러지 관련 reward구하는 matrix 연산자 구현하기 
start = time.time()  # 시작 시간 저
for i in range(len(list(tf_dataset))):

    tmp_diet = list(tf_dataset)[i] # 식단 배치
    tmp_diet = tf.cast(tmp_diet, dtype = tf.int64) # int 타입으로

    # onehot 벡터인데 time_step이 없는 one_hot으로 -> 발생빈도 벡터가 됨
    tmp_diet = tf.reduce_sum(tf.one_hot(tmp_diet, depth = len(food_dict)), axis = 1)
    tmp_diet_np = tmp_diet.numpy()
    tmp_diet_np[tmp_diet_np >= 1] = 1   # 발생빈도가 1이상이면 1로

    # 알러지 유발메뉴 포함 식단의 인덱스 (포함 = 1, 비포함 = 0)
    alergy_diets_idx = np.where(np.dot(tmp_diet_np, alergy_menu_vector).flatten() == 1)[0]  

    # aa는 알러지 유발메뉴 = 0, 비유발메뉴 = 1로하는, 전체 식단 샘플갯수만큼의 길이를 각진 onehot 벡터가 됨
    aa = np.ones(len(tf_dataset))
    aa[alergy_diets_idx] = 0
    aaa = aa.reshape(-1, 1)     # aaa는 aa를 컬럼벡터로 만든 것.
    
print("time :", time.time() - start)

diet_data.iloc[np.where(np.dot(tmp_diet_np, alergy_menu_vector) == 0)[0], :]

# %%
## get_reward_ver2를 효율적으로 하는 방법은 map인가 apply_along_with_axis 인가?
start = time.time()  # 시작 시간 저장
print(np.array(list(map(lambda x: get_reward_ver2(x, done = 0), nutrient_score))))
print("time :", time.time() - start)

start = time.time()  # 시작 시간 저장
print(np.apply_along_axis(get_reward_ver2, axis = 1, arr = nutrient_score, done = 0))
print("time :", time.time() - start)

# %%
## 뉴럴넷의 예측 토큰이 무엇인지 확률이 큰 순으로 확인해보기
t = 0
x = list(tf_dataset)[0]
enc_hidden = encoder.initialize_hidden_state()
inputs = x[:, :x.shape[1] - 1]
targets = x[:, 1:x.shape[1]]
train_seqs = tf.reshape(inputs[:, 0], shape = (-1, 1)) 
enc_output, enc_hidden = encoder(inputs, enc_hidden)
dec_hidden = copy.deepcopy(enc_hidden)
preds, dec_hidden, _ = decoder(train_seqs[:, t], dec_hidden, enc_output)
greedy_results = np.apply_along_axis(get_action, axis = 1, arr = preds, option = 'prob')
greedy_actions = tf.reshape(greedy_results[:, 0], shape = (-1, 1))
food_dict[greedy_actions[0].numpy().astype(int)[0]]   
tmp = [food_dict[i] for i in np.flip(np.argsort(preds[0]))[:50]]


# %%
## 데이터 EDA
tmp = np.apply_along_axis(get_score_vector, arr = diet_data_np, axis = 1, nutrient_data = nutrient_data)
rewardss = np.apply_along_axis(get_reward_ver2, arr = tmp, axis = 1, done = 0)[:, 0]
rewardss_type = np.apply_along_axis(get_reward_ver2, arr = tmp, axis = 1, done = 0)[:, 2]
sorted_rewardss = rewardss[np.argsort(rewardss)]
min_freq_reward = np.min(sorted_rewardss)
fig = plt.figure(tight_layout = True)
plt.bar(np.unique(sorted_rewardss), np.bincount(sorted_rewardss.astype(int))[min_freq_reward:])
plt.xticks(np.unique(sorted_rewardss).tolist())
plt.xlabel('reward')
plt.ylabel('freq')
fig.patch.set_facecolor('white')
figure_dir = '/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/figures/reward_distributoin.png'
plt.savefig(figure_dir, bbox_inches='tight')

## 영양기준 분포
rewardss_mat = np.reshape(np.stack(rewardss_type, axis = 0), newshape = (-1, 13))
rewards_distribution = np.sum(rewardss_mat, axis = 0)
a = np.array(['calorie', 'protein', 'fiber', 'vitaA', 'vitaC', 'vitaB1', 'vataB2', 'calcium', 'iron', 'sodium', 'lino', 'alpha', 'macroRatio'])
plt.bar(a, rewards_distribution)
plt.xticks(rotation = 45)
plt.show()
reward_type = np.array(['Energy', 'Protein(g)', 'Fiber(g)', 'VitaA', 'VitaC', 'VitaB1', 'VitaB2', 'Calcium', 'Iron', 'Sodium', 'Linoleic', 'Alpha-Linolen','macroRatio'])


## 최고점짜리 식단들
suboptimal_diet = diet_data_np[np.where(rewardss == np.max(sorted_rewardss))[0], :]
best_diet = sequence_to_sentence(suboptimal_diet, food_dict)
best = np.apply_along_axis(get_score_vector, arr = suboptimal_diet, axis = 1, nutrient_data = nutrient_data)
best_type = np.apply_along_axis(get_reward_ver2, arr = best, axis = 1, done = 0)[:, 2]
pd.DataFrame(best_diet).to_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data/best_diet.csv', encoding="utf-8-sig")

best_labels = reward_type[np.argsort(np.stack(best_type, axis = 0).sum(axis = 0))]
best_values = np.stack(best_type, axis = 0).sum(axis = 0)[np.argsort(np.stack(best_type, axis = 0).sum(axis = 0))]
plt.bar(best_labels, best_values)
plt.xticks(rotation = 45)
plt.title('best_diet_case')
plt.show()

## 최저점짜리 식단들
suboptimal_diet = diet_data_np[np.where(rewardss == np.min(sorted_rewardss))[0], :]
worst_diet = sequence_to_sentence(suboptimal_diet, food_dict)
pd.DataFrame(worst_diet).to_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data/worst_diet.csv', encoding="utf-8-sig")
worst = np.apply_along_axis(get_score_vector, arr = suboptimal_diet, axis = 1, nutrient_data = nutrient_data)
worst_type = np.apply_along_axis(get_reward_ver2, arr = worst, axis = 1, done = 0)[:, 2]

worst_labels = reward_type[np.argsort(np.stack(worst_type, axis = 0).sum(axis = 0))]
worst_values = np.stack(worst_type, axis = 0).sum(axis = 0)[np.argsort(np.stack(worst_type, axis = 0).sum(axis = 0))]
plt.bar(worst_labels, worst_values)
plt.xticks(rotation = 45)
plt.title('worst_diet_case')
plt.show()

## 통계량
df_tmp = pd.DataFrame(tmp)
df_tmp.columns = nutrient_data.columns[1:] 
df_tmp.to_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data/diet_nutrition.csv', encoding="utf-8-sig")
statistics = df_tmp.describe()
statistics.to_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data/statistics.csv', encoding="utf-8-sig")


# %%
'''
줄리아 생성 결과 전처리
'''
tmp_julia_result = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/Baselines/MP_Solver/gen_menu_mat2.csv', header = None)
julia_result = tmp_julia_result.apply(lambda x: x.str.replace(r"[^\d]", ""), axis = 1)

aaaa = np.array(julia_result).astype(int) - 1
get_score_vector(aaaa.flatten(), nutrient_data)
pd.DataFrame(sequence_to_sentence(np.array(julia_result).astype(int) - 1, food_dict))
 
julia_diets = pd.DataFrame(sequence_to_sentence(np.array(julia_result).astype(int) - 1, food_dict)).T

julia_diets.to_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/results/julia_result.csv', encoding="utf-8-sig")


# %%
'''
병렬처리
'''
from multiprocessing import Pool
import time
import os
import math
import numpy as np

def f(x):
    proc = os.getpid()
    number = x
    result = sum(x)
    print('{0} summed to {1} by process id: {2}'.format(number, result, proc))
    # time.sleep(1)
    return result

p = Pool(3)
# dat = [5, 10, 15, 20, 25]
dat = np.array([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 9]])
startTime = time.time()
print(p.map(f, dat))  # 함수와 인자값을 맵핑하면서 데이터를 분배한다
endTime = time.time()
print("총 작업 시간", (endTime - startTime))

from multiprocessing import Process

def doubler(number):
    # A doubling function that can be used by a process
    
    result = number * 2
    proc = os.getpid()
    print('{0} doubled to {1} by process id: {2}'.format(
        number, result, proc))

numbers = [5, 10, 15, 20, 25]
procs = []

startTime = time.time()
for index, number in enumerate(numbers):
    proc = Process(target=doubler, args=(number,))
    procs.append(proc)
    proc.start()

for proc in procs:
    proc.join()
endTime = time.time()
print("총 작업 시간", (endTime - startTime))
