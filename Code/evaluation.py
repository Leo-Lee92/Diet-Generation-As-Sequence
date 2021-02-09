# %%
import sys
sys.path.append("/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code")
from Preprocessing import *
from util import *
from Model import *
import argparse
parser = argparse.ArgumentParser(description='model,  lr 입력')
parser.add_argument('--num_epochs', type=int, required=True, help='num_epochs 입력')
parser.add_argument('--lr', type=float, required=True, help='learning_rate 입력')
args = parser.parse_args()

# %%
'''
TSNE Mapping
'''

## tsne map with generated diets 그리기 (복수의 방법론 비교)
from os import listdir
from os.path import isfile, join
target_dir_list = [path + '/' + subdir for (path, dir, files) in os.walk('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code') for subdir in dir if "tsne" in subdir]
total_nutrient_df = []
method_label = []
reward_dist_stack = np.empty([0, 13])
np.random.seed(None)

for cp_dir in target_dir_list:

    # 사전학습 모델을 활용하여 강화학습
    # pretrain 폴더에서 특정 시점 체크포인트 복원하기
    # 변수 초기화를 위한 random seed로서의 input, hidden_state, concat_state 생성
    encoder = Encoder(len(food_dict), BATCH_SIZE)
    init_input = np.zeros([BATCH_SIZE, 1])
    init_hidden = encoder.initialize_hidden_state()
    init_output, _ = encoder(init_input, init_hidden)

    # Decoder to predict food sequence
    decoder = Decoder(len(food_dict))
    decoder(init_input, init_hidden, init_output)

    # (1-3) 체크포인트에 기록된 인스턴스 지정
    checkpoint = tf.train.Checkpoint(encoder = encoder, decoder = decoder)
    # checkpoint.restore(tf.train.latest_checkpoint("/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/Baselines/SCST/pretraining_SCST"))
    # checkpoint.restore(tf.train.latest_checkpoint("/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/Baselines/SCST/training_SCST_lr=0.001_epoch=100/0"))
    # checkpoint.restore(tf.train.latest_checkpoint("/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/Baselines/MIXER/training_lr=0.001_epoch=1000"))
    # checkpoint.restore(tf.train.latest_checkpoint("/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/Proposed/training_lr=0.001_epoch=5000_eb=10_bs=10"))
    checkpoint.restore(tf.train.latest_checkpoint(cp_dir))

    encoder = checkpoint.encoder
    decoder = checkpoint.decoder

    true_total_reward = 0
    gen_total_reward = 0
    # total_nutrient_df_real = pd.DataFrame()
    total_nutrient_df_gen = pd.DataFrame()

    for batch in range(len(list(tf_dataset))):
        x = list(tf_dataset)[batch]
        sample_input = x[:, :x.shape[1] - 1]
        sample_enc_hidden = encoder.initialize_hidden_state()

        # 두 인코더의 컨텍스트 벡터 각각 뽑아주기
        sample_enc_output, sample_enc_hidden = encoder(sample_input, sample_enc_hidden)

        # 두 인코더의 컨텍스트 벡터 연결해주기
        sample_dec_hidden = copy.deepcopy(sample_enc_hidden)

        sample_seqs = np.empty((0, 1))
        sample_seqs = np.concatenate([sample_seqs, tf.reshape(sample_input[:, 0], shape = (-1, 1))])

        for j in range(15):
            sample_outputs, sample_dec_hidden, attention_weigths = decoder(sample_input[:, j], sample_dec_hidden, sample_enc_output)
            results = np.apply_along_axis(get_action, axis = 1, arr = sample_outputs, option = 'prob')
            next_token = tf.reshape(results[:, 0], shape = (-1, 1))
            sample_seqs = np.concatenate([sample_seqs, next_token], axis = 1)

        generated_file_name = "/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/results/" + cp_dir.split('_')[-2] + "_data_np.csv"
        pd.DataFrame(sample_seqs).to_csv(generated_file_name)

        # 실제 식단과 생성 식단의 영양상태를 저장하여 t-sne 맵 만들기
        nutrient_real = np.apply_along_axis(get_score_vector, axis = 1, arr = np.array(x), nutrient_data = nutrient_data)
        nutrient_gen = np.apply_along_axis(get_score_vector, axis = 1, arr = sample_seqs, nutrient_data = nutrient_data)

        # total_nutrient_df_real = total_nutrient_df_real.append(pd.DataFrame(nutrient_real))
        total_nutrient_df_gen = total_nutrient_df_gen.append(pd.DataFrame(nutrient_gen))

        print(' ')
        print(' 정답 :', sequence_to_sentence(np.array(x), food_dict)[0])
        print(' 생성 :', sequence_to_sentence(sample_seqs, food_dict)[0])

        generated_file_name = "/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/results/" + cp_dir.split('_')[-2] + "_result.csv"
        pd.DataFrame(sequence_to_sentence(sample_seqs, food_dict)).to_csv(generated_file_name, encoding = 'utf-8-sig')

        true_reward = get_reward_ver2(get_score_vector(x[0], nutrient_data), 0)[0]
        gen_reward = get_reward_ver2(get_score_vector(sample_seqs[0], nutrient_data), 0)[0]
        print(' ')
        print(' 정답의 보상 :', true_reward)
        print(' 생성의 보상 :', gen_reward)

        mean_true_reward = np.mean(np.apply_along_axis(get_reward_ver2, axis = 1, arr = nutrient_real, done = 0)[:, 0])
        mean_gen_reward = np.mean(np.apply_along_axis(get_reward_ver2, axis = 1, arr = nutrient_gen, done = 0)[:, 0])

        # 배치가 여러개 일 때 누적
        true_total_reward += mean_true_reward
        gen_total_reward += mean_gen_reward

        # 생성식단들의 reward
        reward_dist = np.apply_along_axis(get_reward_ver2, axis = 1, arr = nutrient_gen, done = 0)[:, 2].sum(axis = 0).reshape(1, -1)
        reward_dist_stack = np.append(reward_dist_stack, reward_dist, axis = 0)

    true_mean_reward = true_total_reward / len(list(tf_dataset))
    gen_mean_reward = gen_total_reward / len(list(tf_dataset))

    print('true_mean_reward :', true_mean_reward)
    print('gen_mean_reward :', gen_mean_reward)

    total_nutrient_df, method = make_matrix_for_tsne(total_nutrient_df_gen)

# tsne 매핑 결과 보기
total_nutrient_df_real = pd.DataFrame(nutrient_real)
tsne_matrix = tsne_mapping_multiple_gen(total_nutrient_df_real, total_nutrient_df)
tsne_plot(tsne_matrix, method)

# %%
'''
Hit rate 점수 책정
'''
first_element = np.where(np.array(list(food_dict.values())) == "시작")[0][0]
last_element = np.where(np.array(list(food_dict.values())) == "종료")[0][0]

# 1) 실제
# meal_hit_score(diet_data_np.astype('int'), category_data)
# dish_hit_score(diet_data_np.astype('int'), category_data)
avg_real_meal_hit = np.stack(np.apply_along_axis(meal_hit_score, axis = 1, arr = diet_data_np.astype('int'), category_data = category_data)[:, 0], axis = 0).mean()
avg_real_dish_hit = np.stack(np.apply_along_axis(dish_hit_score, axis = 1, arr = diet_data_np.astype('int'), category_data = category_data)[:, 0], axis = 0).mean()

# 2) MIP
# MIP_result = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/results/julia_result.csv', index_col = 0)
MIP_result = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/results/available_julia.csv', index_col = 0)
MIP_data_np = food_to_token(MIP_result, nutrient_data)
first_column = np.repeat(first_element, MIP_data_np.shape[0]).reshape(-1, 1)
last_column = np.repeat(last_element, MIP_data_np.shape[0]).reshape(-1, 1)
MIP_data_np = np.hstack([first_column, MIP_data_np, last_column])

# meal_hit_score(MIP_data_np.astype('int'), category_data)
# dish_hit_score(MIP_data_np.astype('int'), category_data)
avg_MIP_meal_hit = np.stack(np.apply_along_axis(meal_hit_score, axis = 1, arr = MIP_data_np.astype('int'), category_data = category_data)[:, 0], axis = 0).mean()
avg_MIP_dish_hit = np.stack(np.apply_along_axis(dish_hit_score, axis = 1, arr = MIP_data_np.astype('int'), category_data = category_data)[:, 0], axis = 0).mean()

# 3) SCST
SCST_result = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/results/SCST_data_np.csv', index_col = 0)
SCST_data_np = np.array(SCST_result)

# SCST_data_np = food_to_token(SCST_result, nutrient_data)
# first_column = np.repeat(first_element, SCST_data_np.shape[0]).reshape(-1, 1)
# last_column = np.repeat(last_element, SCST_data_np.shape[0]).reshape(-1, 1)
# SCST_data_np = np.hstack([first_column, SCST_data_np, last_column])

# meal_hit_score(SCST_data_np.astype('int'), category_data)
# dish_hit_score(SCST_data_np.astype('int'), category_data)
avg_SCST_meal_hit = np.stack(np.apply_along_axis(meal_hit_score, axis = 1, arr = SCST_data_np.astype('int'), category_data = category_data)[:, 0], axis = 0).mean()
avg_SCST_dish_hit = np.stack(np.apply_along_axis(dish_hit_score, axis = 1, arr = SCST_data_np.astype('int'), category_data = category_data)[:, 0], axis = 0).mean()

# 4) MIXER
MIXER_result = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/results/MIXER_data_np.csv', index_col = 0)
MIXER_data_np = np.array(MIXER_result)

# MIXER_data_np = food_to_token(MIXER_result, nutrient_data)
# first_column = np.repeat(first_element, MIXER_data_np.shape[0]).reshape(-1, 1)
# last_column = np.repeat(last_element, MIXER_data_np.shape[0]).reshape(-1, 1)
# MIXER_data_np = np.hstack([first_column, MIXER_data_np, last_column])

# meal_hit_score(MIXER_data_np.astype('int'), category_data)
# dish_hit_score(MIXER_data_np.astype('int'), category_data)
avg_MIXER_meal_hit = np.stack(np.apply_along_axis(meal_hit_score, axis = 1, arr = MIXER_data_np.astype('int'), category_data = category_data)[:, 0], axis = 0).mean()
avg_MIXER_dish_hit = np.stack(np.apply_along_axis(dish_hit_score, axis = 1, arr = MIXER_data_np.astype('int'), category_data = category_data)[:, 0], axis = 0).mean()

# 5) TFR
TFR_result = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/results/TFR_data_np.csv', index_col = 0)
TFR_data_np = np.array(TFR_result)

# TFR_data_np = food_to_token(TFR_result, nutrient_data)
# first_column = np.repeat(first_element, TFR_data_np.shape[0]).reshape(-1, 1)
# last_column = np.repeat(last_element, TFR_data_np.shape[0]).reshape(-1, 1)
# TFR_data_np = np.hstack([first_column, TFR_data_np, last_column])

# meal_hit_score(TFR_data_np.astype('int'), category_data)
# dish_hit_score(TFR_data_np.astype('int'), category_data)
avg_TFR_meal_hit = np.stack(np.apply_along_axis(meal_hit_score, axis = 1, arr = TFR_data_np.astype('int'), category_data = category_data)[:, 0], axis = 0).mean()
avg_TFR_dish_hit = np.stack(np.apply_along_axis(dish_hit_score, axis = 1, arr = TFR_data_np.astype('int'), category_data = category_data)[:, 0], axis = 0).mean()

# %%
'''
RDI score 점수 책정
'''
real_score = np.apply_along_axis(get_score_vector, arr = diet_data_np, axis = 1, nutrient_data = nutrient_data)
real_RDI_score = np.apply_along_axis(get_reward_ver2, arr = real_score, axis = 1, done = 0)[:, 0].mean()

'''
여기에서 MIP는 지금 보상이 잘못 계산되고 있음을 유념하라.
>> 내가 작성한 보상을 계산하는 함수는 index 기반이라 nutrient_data['energy']가 아니라 nutrient_data.iloc(axis)[1] 와 같이 각 영양소를 조회함.
>> 그러나 MIP로 생성한 식단은 순서가 지멋대로이므로 이렇게 index기반으로 계산할 시 잘못된 보상을 계산하게 됨.
'''
MIP_score = np.apply_along_axis(get_score_vector, arr = MIP_data_np, axis = 1, nutrient_data = nutrient_data)
MIP_RDI_score = np.apply_along_axis(get_reward_ver2, arr = MIP_score, axis = 1, done = 0)[:, 0].mean()

SCST_score = np.apply_along_axis(get_score_vector, arr = SCST_data_np, axis = 1, nutrient_data = nutrient_data)
SCST_RDI_score = np.apply_along_axis(get_reward_ver2, arr = SCST_score, axis = 1, done = 0)[:, 0].mean()

MIXER_score = np.apply_along_axis(get_score_vector, arr = MIXER_data_np, axis = 1, nutrient_data = nutrient_data)
MIXER_RDI_score = np.apply_along_axis(get_reward_ver2, arr = MIXER_score, axis = 1, done = 0)[:, 0].mean()

TFR_score = np.apply_along_axis(get_score_vector, arr = TFR_data_np, axis = 1, nutrient_data = nutrient_data)
TFR_RDI_score = np.apply_along_axis(get_reward_ver2, arr = TFR_score, axis = 1, done = 0)[:, 0].mean()


# %%
'''
Data Merge for Barplotting
'''
real_reward = np.apply_along_axis(get_reward_ver2, axis = 1, arr = nutrient_real, done = 0)[:, 2].sum(axis = 0).reshape(1, -1)
MIP_reward = np.ones([1, 13]) * 1072    # MIP의 결과는 만점이므로 그냥 모든 영양소 기준에 대해 만족한 식단의 갯수를 1072로 고정

reward_dist_stack = np.append(real_reward, reward_dist_stack, axis = 0)   # 방법론 별 보상 분포
reward_dist_stack = np.append(reward_dist_stack, MIP_reward, axis = 0)   # 방법론 별 보상 분포

types_of_nutrient = np.array([['calorie', 'protein', 'fiber', 'vitaA', 'vitaC', 'vitaB1', 'vitaB2', 'calcium', 'iron', 'sodium', 'linoleic', r'$\alpha$-linolenic', 'macroRatio']]).T
types_of_method = ['real', 'TFR', 'SCST', 'MIXER', 'MIP']

conat_rds_all = np.empty([0, 3]) # value, nutrient, method 3개 축으로 구성됨
for i in range(reward_dist_stack.shape[0]):
    concat_rds = np.append(reward_dist_stack[i, :].reshape(-1, 1), types_of_nutrient, axis = 1)
    concat_rds = np.append(concat_rds, np.repeat(types_of_method[i], concat_rds.shape[0]).reshape(-1, 1), axis = 1)
    conat_rds_all = np.vstack([conat_rds_all, concat_rds])

reward_dist_stack_df = pd.DataFrame(conat_rds_all)
reward_dist_stack_df.columns = np.array(['count', 'nutrient', 'method'])
reward_dist_stack_df['count'] = reward_dist_stack_df['count'].astype('float')
reward_dist_stack_df['nutrient'] = reward_dist_stack_df['nutrient'].astype('str')

# %%
'''
Barplotting
'''
colors = ["#d62728", "#1f77b4"]
# sns.set(style = 'darkgrid')
sns.set_palette(sns.color_palette(colors))

nut_ = reward_dist_stack_df[(reward_dist_stack_df['method'] == "real") | (reward_dist_stack_df['method'] == "TFR")]
nut_list = [nut_]
plt.xticks(rotation = 45)

# colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "black"]
# nut0 = reward_dist_stack_df[(reward_dist_stack_df['nutrient'] == "calorie") | (reward_dist_stack_df['nutrient'] == "protein") | (reward_dist_stack_df['nutrient'] == "macroRatio")]
# nut1 = reward_dist_stack_df[(reward_dist_stack_df['nutrient'] == "vitaA") | (reward_dist_stack_df['nutrient'] == "vitaC") | (reward_dist_stack_df['nutrient'] == "vitaB1") | (reward_dist_stack_df['nutrient'] == "vitaB2")]
# nut2 = reward_dist_stack_df[(reward_dist_stack_df['nutrient'] == "fiber") | (reward_dist_stack_df['nutrient'] == "calcium") | (reward_dist_stack_df['nutrient'] == "iron") | (reward_dist_stack_df['nutrient'] == "sodium")]
# nut3 = reward_dist_stack_df[(reward_dist_stack_df['nutrient'] == "linoleic") | (reward_dist_stack_df['nutrient'] == r"$\alpha$-linolenic")]
# nut_list = [nut0, nut1, nut2, nut3]

for i, val in enumerate(nut_list):
    plot_name = "reward_distributoin_compare_nut_.png"
    # plot_name = "reward_distributoin_compare_nut_" + str(i) + ".png"
    ax = sns.barplot(x = 'nutrient', y = 'count', hue = 'method', data = val)

    for p in ax.patches:
        left, bottom, width, height = p.get_bbox().bounds
        ax.annotate("%.1d" % height, (left + width/2, height), ha='center', va='center', fontsize = 7, rotation = 90, xytext = (0, 10), textcoords = 'offset points')

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    figure_dir = '/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/figures/' + plot_name
    plt.savefig(figure_dir, bbox_inches='tight')
    plt.clf()
# %%
'''
Reward Ploting
'''

## training plot over time-through rewards 그리기
from os import listdir
from os.path import isfile, join

# 단일 방법론
# dir_file_name = '/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/results/SCST_eb=None_bs=None_lr=0.001_epoch=5000_rewards.csv'
# plot_reward(dir_file_name)

# 복수의 방법론 비교
# reward plot 그릴 파일 담기
files = [f for f in listdir('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/results') if isfile(join('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/results', f))]
# keyword1 = "SCST"
# keyword2 = "MIXER"
# keyword3 = "bs=10"
# filtered_files = [a_file for a_file in files if keyword1 in a_file or keyword2 in a_file or keyword3 in a_file]
keyword = "plot"
filtered_files = [a_file for a_file in files if keyword in a_file]

# filtered_files 리스트에 담긴 파일 순서 바꾸기
my_order = [1, 2, 0]
rearranged_files = [filtered_files[order] for order in my_order] 

# results로 현재 디렉토리 (cwd) 디렉토리 변경 (change directory: chdir)
os.chdir("/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/results")

trunc_range = 1
reward_over_methods_df = pd.DataFrame([])
for each_file in rearranged_files:
    method = each_file.split('_eb')[0]
    tmp = pd.read_csv(each_file)
    tmp['method'] = np.repeat(method, tmp.shape[0])

    # observation을 trunc_range로 잘라서 각 range별로 reward를 보고 싶다면
    # tmp2 = tmp.iloc(axis = 0)[tmp['reward'].groupby(tmp['reward'].index // trunc_range).agg(['min', 'median', 'mean', 'max']).dropna().index * trunc_range + (trunc_range - 1)]
    # tmp2.reset_index(inplace = True)
    # reward_over_methods_df = pd.concat([reward_over_methods_df, tmp2])

    # 순전하게 모든 observation에서의 reward를 보고 싶다면
    reward_over_methods_df = pd.concat([reward_over_methods_df, tmp])

reward_over_methods_df.to_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/results/total_rewards.csv')

plot_reward('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/results/total_rewards.csv')
# %%
