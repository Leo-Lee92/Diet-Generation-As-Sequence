# %%
'''
Generation
'''
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import tensorflow as tf
import dietkit
from util import sequence_to_sentence, get_TargetDir, loadParams, kor_to_eng, translate_dict, get_score_pandas, get_reward_ver2
from Model import Sequence_Generator

# (step 0) Get data
source_diet_seqs = np.array(pd.read_csv('./results/diet_seqs.csv', index_col = 0))
BATCH_SIZE = source_diet_seqs.shape[0]
tf_dataset_for_eval = tf.cast(source_diet_seqs, dtype = tf.int32)                   # Make source diet as tensor type
tf_dataset_for_eval = tf.data.Dataset.from_tensor_slices(tf_dataset_for_eval)       # Transform numpy object to tensorflow Dataset object and slices it into tensor-like form (i.e., define the number of dim and set the axis).
tf_dataset_for_eval = tf_dataset_for_eval.batch(BATCH_SIZE, drop_remainder = True)  # Make batch according to BATCH_SIZE.

nutrient_data = pd.read_csv('./results/nutrient_data.csv', index_col = 0)

# (step 1) Parameter Intialization
## Initialize the variable to store nutrient vectors of generated diets.
total_nutrient_df = []

## Initialize the variable to store method labels.
method_label = []

## Initialize the variable to stack reward distributions
reward_dist_stack = np.empty([0, 15])   # Here, the number 13 means the number of rewards considered.
np.random.seed(None)

## Initialize the number of batches
batch_num = len(list(tf_dataset_for_eval))               

## Initialize the length of sequence without 'bos' and 'eos'
seq_len = list(tf_dataset_for_eval)[0].shape[1] - 2

# (step 2) 
# Extract every leaf directories (i.e., subdir) with given parameters (e.g., 'epoch=10000_rs=True') and stemed from the node directory '/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code'. Then, store the full name of subdir (i.e, path + '/' + subdir) into target_dir_list.
target_address = 'params'
target_keyword = 'with_attention'
target_dir_list = get_TargetDir(target_address, target_keyword)

# (step 3) Visualize the results of checkpoints saved in target_dir_list.
k = 3    # 임시
target_dir = target_dir_list[k] 
print('target_dir :', target_dir)

## --- (1) Define the required parameters, variables, models etc.
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

## --- (6) Run prediction using pretrained parameters of the restored model.
for batch in range(batch_num):

    # Define a single batch sample.
    x = list(tf_dataset_for_eval)[batch]                 

    # Generate sequences (i.e., synthetic diets).
    gen_seqs, _ = diet_generator.inference(x)       
    
    # Check the reward of real and generated diets in first index.
    print('In korean')
    print(' 정답 :', sequence_to_sentence(x.numpy(), food_dict)[0])
    print(' 생성 :', sequence_to_sentence([gen_seqs[0]], food_dict)[0])
    print(' ')
    print('In english')
    print(' 정답 :', kor_to_eng([sequence_to_sentence(x.numpy()[:, 1:x.shape[1] - 1], food_dict)[0]], translate_dict) )
    print(' 생성 :', kor_to_eng([sequence_to_sentence(gen_seqs[:, 1:x.shape[1] - 1], food_dict)[0]], translate_dict) )

# %%
# Save the generated result
# 생성결과 저장
## korean version
pd.DataFrame(kor_to_eng(sequence_to_sentence(gen_seqs, food_dict), translate_dict)).to_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/results/gen_diets(eng).csv', encoding='CP949')
## english version
pd.DataFrame(sequence_to_sentence(gen_seqs, food_dict)).to_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/results/gen_diets(kor).csv', encoding='CP949')

'''
Evaluation with RDI score
'''
# real
expert_scores = get_score_pandas(x.numpy(), food_dict, nutrient_data)
RDI_score_real = np.apply_along_axis(get_reward_ver2, arr = expert_scores, axis = 1, done = 0)[:, 0]
print('RDI_score of expert:', RDI_score_real.mean())

# gen
ML_scores = get_score_pandas(gen_seqs, food_dict, nutrient_data)
RDI_score_gen = np.apply_along_axis(get_reward_ver2, arr = ML_scores, axis = 1, done = 0)[:, 0]
print('RDI_score of ML:', RDI_score_gen.mean())

'''
Visulaize Nutrients Distribution
'''
# expert
expert_scores = get_score_pandas(x.numpy(), food_dict, nutrient_data)
num_achieved_nutrients_expert = np.apply_along_axis(get_reward_ver2, arr = expert_scores, axis = 1, done = 0)[:, 2]
nutri_ratio_expert = num_achieved_nutrients_expert.mean()

# ML
ML_scores = get_score_pandas(gen_seqs, food_dict, nutrient_data)
num_achieved_nutrients_ML = np.apply_along_axis(get_reward_ver2, arr = ML_scores, axis = 1, done = 0)[:, 2]
nutri_ratio_ML = num_achieved_nutrients_ML.mean()
 
achievement_ratios = pd.DataFrame([nutri_ratio_expert, nutri_ratio_ML]).T
achievement_ratios.columns = ['expert', 'ML']
achievement_ratios.index = ['energy', 'protein', 'fiber', 'vitaA', 'vitaC', 'vitaB1', 'vitaB2', 'calcium', 'iron', 'sodium', 'linoleic', r'$\alpha$-linolenic', '(%) carbo', '(%) protein', '(%) fat']
achievement_ratios.plot.bar(rot = 90)
plt.title('Achievement Rates per Nutrient')
plt.savefig('./figures/achievement_ratios.png', dpi=300, bbox_inches='tight')


# # %%
# # Import dietkit
# from dietkit.loader import *

# # Load menus
# menus_dict = load_menu()

# # expert
# # Get samples of generated diet.
# ## get sample without BOS and EOS tokens.
# sample_diets_expert = kor_to_eng(sequence_to_sentence(x.numpy()[:, 1:x.shape[1] - 1], food_dict))

# ## process string error.
# ### (1) make all menus as lowercase 
# ### (2) remove blank in diet
# ### (3) make first letter as capital letter
# ### (4) but menu 'empty' is exception.
# for diet_idx, diet in enumerate(sample_diets_expert):
#     for menu_idx, menu in enumerate(diet): 
#         if diet[menu_idx] != "empty":
#             diet[menu_idx] = menu.lower().strip().capitalize()
#     sample_diets_expert[diet_idx] = diet

# ## change diet as object_diet which is an object-type of diet addressed by dietkit, otherwise you cannot use dietkit.
# object_diet_expert= []
# for diet_idx, diet in enumerate(sample_diets_expert):
#     object_diet_expert.append([ menus_dict[menu] for menu in diet ])   

# # Get ingredient of diet
# object_diet_expert = Diet(dict(list(zip(range(len(object_diet_expert)), object_diet_expert[:]))))

# # ML
# # Get samples of generated diet.
# ## get sample without BOS and EOS tokens.
# sample_diets_ML = kor_to_eng(sequence_to_sentence(gen_seqs[:, 1:gen_seqs.shape[1] - 1], food_dict))

# ## process string error.
# ### (1) make all menus as lowercase 
# ### (2) remove blank in diet
# ### (3) make first letter as capital letter
# ### (4) but menu 'empty' is exception.
# for diet_idx, diet in enumerate(sample_diets_ML):
#     for menu_idx, menu in enumerate(diet): 
#         if diet[menu_idx] != "empty":
#             diet[menu_idx] = menu.lower().strip().capitalize()
#     sample_diets_ML[diet_idx] = diet

# ## change diet as object_diet which is an object-type of diet addressed by dietkit, otherwise you cannot use dietkit.
# object_diet_ML = []
# for diet_idx, diet in enumerate(sample_diets_ML):
#     object_diet_ML.append([ menus_dict[menu] for menu in diet ])   

# # Get ingredient of diet
# object_diet_ML = Diet(dict(list(zip(range(len(object_diet_ML)), object_diet_ML[:]))))

# # %%
# '''
# Analysis with Attention map
# '''
# # Attention Image 뽑는 코드 (evaluation.py로 옮겨주기)
# # Define a single batch sample.
# x = list(tf_dataset_for_eval)[batch]             

# # Generate sequences (i.e., synthetic diets).
# gen_seqs, atts = diet_generator.inference(x, return_attention = True)     

# # Get attention maps of examples where the translated diet has highest distance with source diet. The distance is defined as jaccard distance.
# top_change_sequence = get_most_change_sequence(x, gen_seqs, top_n = 5)
# top_chnage_idx = top_change_sequence['example_idx']
# for i in top_chnage_idx:
#     plot_attention(i, atts[i], x.numpy()[i], gen_seqs[i], food_dict, translate_dict, language = 'eng')

# # %%
# # Apply tsne and visualize the tsne map.
# # tsne 매핑 결과 보기
# total_nutrient_df_real = pd.DataFrame(nutrient_real)    # shape : num_datapoints by 17 (Note. Here, number 17 indicates the number of nutrients we consider.)
# tsne_matrix = tsne_mapping_multiple_gen(total_nutrient_df_real, total_nutrient_df)  # shape : num_datapoints by 2
# tsne_plot(tsne_matrix, method)
