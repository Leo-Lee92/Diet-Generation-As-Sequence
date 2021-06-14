# %%
'''
evaluation
'''
from os import listdir
from os.path import isfile, join
import numpy as np
# (step 1) Parameter Intialization
## Initialize the variable to store nutrient vectors of generated diets.
total_nutrient_df = []

## Initialize the variable to store method labels.
method_label = []

## Initialize the variable to stack reward distributions
reward_dist_stack = np.empty([0, 13])   # Here, the number 13 means the number of rewards considered.
np.random.seed(None)

## Initialize the number of batches
batch_num = len(list(tf_dataset))               

## Initialize the length of sequence without 'bos' and 'eos'
seq_len = list(tf_dataset)[0].shape[1] - 2

# (step 2) Inherit required variables, functions and model from util and Model modules.
from util import *
from Model import Sequence_Generator
from operator import methodcaller

# (step 3) 
# Extract every leaf directories (i.e., subdir) with given parameters (e.g., 'epoch=10000_rs=True') and stemed from the node directory '/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code'. Then, store the full name of subdir (i.e, path + '/' + subdir) into target_dir_list.
target_address = 'params'
target_keyword = 'no_attention'
target_dir_list = get_TargetDir(target_address, target_keyword)

# (step 4) Visualize the results of checkpoints saved in target_dir_list.
for target_dir in target_dir_list:                                                  

    ## --- (1) Define the required parameters, variables, models etc.
    # target_dir = target_dir_list[0] # 임시
    kwargs, food_dict, nutri_dat, inci_dat = loadParams(target_dir)                 # Load all the parameters.
    food_dict = {int(k) : v for k, v in food_dict.items()}                          # Change the key type of dictionary from str to int.
    diet_generator = Sequence_Generator(food_dict, nutri_dat, inci_dat, **kwargs)   # Define sequence_generator.

    ## --- (2) Define checkpoint object which includes encoder and decoder object.
    checkpoint = tf.train.Checkpoint(generator = diet_generator, params = kwargs)

    ## --- (3) Restore checkpoint with the latest object saved at 'target_dir' dirctory.
    cp_dir = target_dir.replace('params', 'checkpoints')                            # Changes directory from '../params' to '../checkpoints' (target_dir -> cp_dir).
    checkpoint.restore(tf.train.latest_checkpoint(cp_dir))                          # Restore checkpoint.

    ## --- (4) Define diet generator restored in checkpoint.
    diet_generator = checkpoint.generator

    ## --- (5) Define varaibles to trace.
    true_total_reward = 0
    gen_total_reward = 0
    # total_nutrient_df_real = pd.DataFrame()
    total_nutrient_df_gen = pd.DataFrame()


    print('target_dir :', target_dir)
    ## --- (6) Run prediction using pretrained parameters of the restored model.
    for batch in range(batch_num):

        # batch = 0 # 임시        

        # Define a single batch sample.
        x = list(tf_dataset)[batch]                 

        # Generate sequences (i.e., synthetic diets).
        gen_seqs = diet_generator.inference(x)       
        
        # Save the generated sequence and its sentence in 'generated_file_name' directory where a file name ends with the suffix '_sequence.csv' and with the suffix '_sentence.csv'. respectively.
        ## The sequence is a vector of tokens to which the number is allocated, while the sentence is a vector that is composed of words corresponding to tokens.
        kwargs_vals = list(map(methodcaller("split", '='), cp_dir.split('/')[-2].split('--')))
        kwargs_vals = np.array(kwargs_vals)[:, 1]
        kwargs_vals = np.append(np.array(cp_dir.split('/')[-4:-2]), kwargs_vals)
        generated_file_name = "/results/" + '--'.join(kwargs_vals) + "_sequence.csv"        
        pd.DataFrame(gen_seqs).to_csv(generated_file_name)
        generated_file_name = "/results/" + '--'.join(kwargs_vals) + "_sentence.csv"
        pd.DataFrame(sequence_to_sentence(gen_seqs, food_dict)).to_csv(generated_file_name, encoding = 'utf-8-sig')

        # Check the reward of real and generated diets in first index.
        print(' ')
        print(' 정답 :', sequence_to_sentence(np.array(x), food_dict)[0])
        print(' 생성 :', sequence_to_sentence(gen_seqs, food_dict)[0])

        true_reward = get_reward_ver2(get_score_vector(x[0], nutrient_data), done = 0, mode = kwargs['add_breakfast'])[0]
        gen_reward = get_reward_ver2(get_score_vector(gen_seqs[0], nutrient_data), done = 0, mode = kwargs['add_breakfast'])[0]
        print(' ')
        print(' 정답의 보상 :', true_reward)
        print(' 생성의 보상 :', gen_reward)

        # Calculate the (nutrient) score of the real and generated diets.
        nutrient_real = np.apply_along_axis(get_score_vector, axis = 1, arr = np.array(x), nutrient_data = nutrient_data)
        nutrient_gen = np.apply_along_axis(get_score_vector, axis = 1, arr = gen_seqs, nutrient_data = nutrient_data)

        # Get reward-related information of true and generated diet sequences.
        reward_info_real = np.apply_along_axis(get_reward_ver2, axis = 1, arr = nutrient_real, done = 0, mode = kwargs['add_breakfast'])
        reward_info_gen = np.apply_along_axis(get_reward_ver2, axis = 1, arr = nutrient_gen, done = 0, mode = kwargs['add_breakfast'])
        
        # Calculate mean rewards of true and generated diet sequences.
        mean_true_reward = np.mean(reward_info_real[:, 0])
        mean_gen_reward = np.mean(reward_info_gen[:, 0])

        # Cumulate the mean rewards of true and generated diet sequences (if there are multiple batches).
        true_total_reward += mean_true_reward
        gen_total_reward += mean_gen_reward

        # Sum the number of achieved reward type over every generated diets and define the variables, named reward_dist, which indicates the distribution of rewards in synthetic data.
        reward_dist = reward_info_gen[:, 2].sum(axis = 0).reshape(1, -1)
        reward_dist_stack = np.append(reward_dist_stack, reward_dist, axis = 0)

        # 실제 식단과 생성 식단의 영양상태를 저장하여 t-sne 맵 만들기
        # total_nutrient_df_real = total_nutrient_df_real.append(pd.DataFrame(nutrient_real))
        total_nutrient_df_gen = total_nutrient_df_gen.append(pd.DataFrame(nutrient_gen))
    
    # Get mean reward per batch
    batch_mean_true_reward = true_total_reward / len(list(tf_dataset))
    batch_mean_gen_reward = gen_total_reward / len(list(tf_dataset))

    print('batch_mean_true_reward :', batch_mean_true_reward)
    print('batch_mean_gen_reward :', batch_mean_gen_reward)

    '''
    TSNE Mapping ('함수화')
    '''
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
MIP_result = pd.read_csv('/results/available_julia.csv', index_col = 0)
MIP_data_np = food_to_token(MIP_result, nutrient_data)
first_column = np.repeat(first_element, MIP_data_np.shape[0]).reshape(-1, 1)
last_column = np.repeat(last_element, MIP_data_np.shape[0]).reshape(-1, 1)
MIP_data_np = np.hstack([first_column, MIP_data_np, last_column])

# meal_hit_score(MIP_data_np.astype('int'), category_data)
# dish_hit_score(MIP_data_np.astype('int'), category_data)
avg_MIP_meal_hit = np.stack(np.apply_along_axis(meal_hit_score, axis = 1, arr = MIP_data_np.astype('int'), category_data = category_data)[:, 0], axis = 0).mean()
avg_MIP_dish_hit = np.stack(np.apply_along_axis(dish_hit_score, axis = 1, arr = MIP_data_np.astype('int'), category_data = category_data)[:, 0], axis = 0).mean()

# 3) SCST
# SCST_result = pd.read_csv('/results/SCST-tsne_data_np.csv', index_col = 0)
SCST_result = pd.read_csv('/results/rs=True-SCST_data_np.csv', index_col = 0)
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
# MIXER_result = pd.read_csv('/results/MIXER-tsne_data_np.csv', index_col = 0)
MIXER_result = pd.read_csv('/results/rs=True-MIXER_data_np.csv', index_col = 0)
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
# TFR_result = pd.read_csv('/results/TFR-tsne_data_np.csv', index_col = 0)
TFR_result = pd.read_csv('/results/rs=True-TFR_data_np.csv', index_col = 0)
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
shapings1 = ['w/o RS', 'w/ RS']
types1 = ['meal', 'dish']
method1 = ['TFR', 'SCST', 'MIXER']

meal_hit1 = [[0.99, 0.78, 0.84, 0.99, 0.68, 0.84]]
dish_hit1 = [[0.96, 0.75, 0.81, 0.96, 0.56, 0.80]]
the_hits = np.append(meal_hit1, dish_hit1)
the_methods = np.tile(method1, 4)
the_shapings = np.tile(np.repeat(shapings1, 3), 2)
the_types = np.repeat(types1, 6)


li = [the_hits, the_methods, the_shapings, the_types]
the_df = pd.DataFrame(li).T
the_df.columns = ['value', 'method', 'shaping', 'hit']

sns.barplot(x = 'method', y = 'value', hue = 'hit', data = the_df[the_df['hit'] == 'meal'])
sns.factorplot(x='shaping', y='value', hue='method', data=the_df, kind='bar')

sns.set(style = 'darkgrid')
sns.factorplot(x='shaping', y='value', hue='method', data=the_df[the_df['hit'] == 'meal'], kind='bar')
plt.xlabel(None)
sns.factorplot(x='shaping', y='value', hue='method', data=the_df[the_df['hit'] == 'dish'], kind='bar')
plt.xlabel(None)

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
    figure_dir = '/figures/' + plot_name
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
# dir_file_name = '/results/SCST_eb=None_bs=None_lr=0.001_epoch=5000_rewards.csv'
# plot_reward(dir_file_name)

# 복수의 방법론 비교
# reward plot 그릴 파일 담기
files = [f for f in listdir('/results') if isfile(join('/results', f))]
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
os.chdir("/results")

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

reward_over_methods_df.to_csv('/results/total_rewards.csv')

plot_reward('/results/total_rewards.csv')
# %%
