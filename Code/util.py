#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import glob
from tensorflow.python.eager.context import LogicalDeviceConfiguration
from tqdm import tqdm
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import seaborn as sns
import copy

'''
transform str to bool.
'''
def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')

'''
preprocessor class that deals with nutrition data.
'''
class nutrition_preprocessor():
    def __init__(self, **preprocessing_kwargs):
        self.feature_data = preprocessing_kwargs['feature_data']

    # insert the features of padding token into nutrition data
    # types of padding token : {empty, bos, eos}
    def insert_padding_feature(self):
        
        # empty token feature
        empty_vector = pd.DataFrame(0, columns = self.feature_data.columns, index = [0])
        empty_vector['name'] = "empty"
        empty_vector['Class'] = "빈칸"
        self.feature_data = pd.concat([empty_vector, self.feature_data]).reset_index(drop = True)

        # bos token feature
        start_vector = pd.DataFrame(0, columns = self.feature_data.columns, index = [0])
        start_vector['name'] = "시작"
        start_vector['Class'] = "앞뒤"
        self.feature_data = pd.concat([self.feature_data, start_vector]).reset_index(drop = True)
    
        # eos token feature
        end_vector = pd.DataFrame(0, columns = self.feature_data.columns, index = [0])
        end_vector['name'] = "종료"
        end_vector['Class'] = "앞뒤"
        self.feature_data = pd.concat([self.feature_data, end_vector]).reset_index(drop = True)

        return self.feature_data

    # Get nutrient-related features from feature data
    def get_nutrient_features(self, feature_data):

        # Call the name of features and use them only to make nutrition data.
        nutrient_feature = list(feature_data.columns.values)
        # nutrient_feature = [e for e in nutrient_feature if e not in ["Weight", "Class", "dish_class1", "dish_class2", "meal_class"]]
        nutrient_feature = [e for e in nutrient_feature if e not in ["Class"]]
        nutrient_feature_data = feature_data.loc(axis = 1)[nutrient_feature]
        # nutrient_feature_data['name'] = nutrient_feature_data['name'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
        nutrient_feature_data['name'] = nutrient_feature_data['name'].str.replace(" ", "")

        return nutrient_feature_data

    def __call__(self):

        # Make new feature dataset where the padding tokens and their corresponding values are inserted.
        new_feature_data = self.insert_padding_feature()

        # Split nutrient feature vector from and make nutrition data
        nutrient_data = self.get_nutrient_features(new_feature_data)

        ## 메뉴 dictionary
        food_dict = dict(new_feature_data['name'])

        return nutrient_data, food_dict
'''
preprocessor class that deals with diet data.
'''
class diet_sequence_preprocessor():
    '''
    나중에 kor, english 구분하는 인자랑 아침포함 diet, 미포함 diet 구분하는 인자 반영해주기
    '''
    def __init__(self, **preprocessing_kwargs):

        # Define global varaibles
        self.diet_data = preprocessing_kwargs['sequence_data']
        self.quality = preprocessing_kwargs['DB_quality']
        self.integrate_DB = preprocessing_kwargs['integrate']

        # Define diet indices initialized by OR, arranged and corrected by expert
        self.or_idx = np.array(range(0, self.diet_data.shape[0], 4))        # diet initialized by OR.
        self.expert_idx1 = np.array(range(1, self.diet_data.shape[0], 4))   # diet arranged by expert.
        self.expert_idx2 = np.array(range(2, self.diet_data.shape[0], 4))   # diet corrected by expert with spec-checker.
        self.expert_idx3 = np.array(range(3, self.diet_data.shape[0], 4))   # diet corrected by expert without spec-checker.

    def select_DB_quality(self):

        if self.quality == 'or':
            diet_data = self.diet_data.iloc[self.or_idx]

        elif self.quality == 'arrange':
            diet_data = self.diet_data.iloc[self.expert_idx1]

        elif self.quality == 'correct1':
            diet_data = self.diet_data.iloc[self.expert_idx2]

        elif self.quality == 'correct2':
            diet_data = self.diet_data.iloc[self.expert_idx3]
            
        return diet_data

    def check_by_nutrient_data(self, diet_data, nutrient_data):

        # Replace possible typo with blank and fill "empty" into the elements that has NaN value.
        # diet_data = diet_data.replace('[^\w]', '', regex=True)
        diet_data.fillna("empty", inplace = True)

        # Get the set of menus used in diet_data
        menus_in_dietDB = set(np.unique( np.char.strip(diet_data.values.flatten().astype('str')) ))                                  
        # Get the set of menus recorded in nutrient_data
        menus_in_nutritionDB = set(np.unique( np.char.strip(nutrient_data['name'].values.flatten().astype('str'))))                  

        # Get the menus which are used in diet data but not exist in nutrient data
        menus_only_in_dietDB = menus_in_dietDB.difference(menus_in_nutritionDB)                                                      

        # print('Total {} menus exist in \'{}\' diet_data'.format(len(menus_in_dietDB), self.quality))
        # print('Total {} menus exist in nutrition_data (menu data)'.format(len(menus_in_nutritionDB)))
        # print('There were {} mismatched menus between \'{}\' diet_data and nutrition_data (menu_data)'.format(len(menus_only_in_dietDB), self.quality))

        if len(menus_only_in_dietDB) > 0:
            # Store the menus that do not exist in nutrient data
            pd.DataFrame(menus_only_in_dietDB).to_csv('../Data/menus_only_in_dietDB.csv', encoding="utf-8-sig")

            # Replace the values with 'empty', that exist in diet_data but not in nutrient_data.
            empty_filled_diet_data = diet_data.replace(menus_only_in_dietDB, 'empty')

            return empty_filled_diet_data
        else:
            return diet_data

    def __call__(self, nutrient_data):

        # if an input diet data consists of multiple source of generation
        if self.integrate_DB == True:
            # Get diet data according to quality which is given by user-defined parameter.
            diet_data = self.select_DB_quality()

        # if an input diet data has a signel source of generation
        else:
            diet_data = self.diet_data

        # Make padding whose the types of padding tokens include 'bos', 'eos', and 'empty', and insert into diet data.
        diet_data.insert(loc = 0, column = "Start", value = ["시작"] * diet_data.shape[0])
        diet_data.insert(loc = diet_data.shape[1], column = "End", value = ["종료"] * diet_data.shape[0])

        # Cross check the mismatched menus using nutrient_data
        diet_data = self.check_by_nutrient_data(diet_data, nutrient_data)

        return diet_data

def createDir(root_dir, params):

    # ------ (3-4) Make sub-directory 1
    if params['attention'] == True:
        subdir1 = 'with_attention'
        target_dir = root_dir + subdir1 + '/'

    else:
        subdir1 = 'no_attention'
        target_dir = root_dir + subdir1 + '/'

    # ------ (3-5) Make sub-directory 2
    subdir2 = params['fully-connected_layer']
    target_dir = target_dir + subdir2 + '/'

    # ------ (3-6) Make final directory
    save_dir = target_dir + '--'.join([i + "=" + str(params[i]) for i in list(params) if i not in ['fully-connected_layer', 'attention', 'num_tokens', 'token_dict']])

    # ------ (3-7) Create a folder of which the name is equal to save_dir and make two sub-directories that store model parameters and its training checkpoints.
    createFolder(save_dir + '/params')
    createFolder(save_dir + '/checkpoints')

    return save_dir

def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('Make a directory at : {}'.format(directory))
    else:
        print('You have {} directory already'.format(directory))            


# Save parameter dictionary as .csv extension.
import csv
import pandas as pd
import numpy as np
import re
def get_TargetDir(target_address, target_keyword):
    # Set root directory
    root_dir = './training_log'

    # Search every subdirectories which contain 'target_address' and define it as 'target_dir_list'.
    target_dir_list = [path + '/' + subdir for (path, dir, files) in os.walk(root_dir) for subdir in dir if target_address in subdir]

    # Compile pattern searcher in terms of pattern 'target_keyword'.
    pattern_searcher = re.compile(target_keyword)

    # Search the compiled pattern in 'target_dir_list' and return indices that contain the pattern.
    target_bool = np.array(list(map(bool, (map(pattern_searcher.search, target_dir_list)))))
    target_idx = np.where(target_bool == True)[0]

    return list(np.array(target_dir_list)[target_idx])

def saveParams(directory, params, dict_dat, feature_dat, inci_dat):
    
    saveInstance = [params, dict_dat, feature_dat, inci_dat]
    nameInstance = ['/kwargs.csv', '/dict_dat.csv', '/feature_dat.csv', '/inci_dat.csv']

    for i in range(len(saveInstance)):
        if type(saveInstance[i]) == dict:
            with open(directory + nameInstance[i], 'w') as file:
                w = csv.writer(file)
                w.writerow(saveInstance[i].keys())
                w.writerow(saveInstance[i].values())

        elif type(saveInstance[i]) == pd.core.frame.DataFrame:
            saveInstance[i].to_csv(directory + nameInstance[i], index = False)

        elif type(saveInstance[i]) == np.ndarray:
            pd.DataFrame(saveInstance[i]).to_csv(directory + nameInstance[i], index = False)

    # with open(directory + '/kwargs.csv', 'w') as file:
    #     w = csv.writer(file)
    #     w.writerow(params.keys())
    #     w.writerow(params.values())

# Load parameter as pandas dataframe format and transform it into dictionary.
import pandas as pd
import tensorflow as tf
def loadParams(directory):
    param_file_dir1 = directory + '/kwargs.csv'
    params_df1 = pd.read_csv(param_file_dir1)

    param_file_dir2 = directory + '/dict_dat.csv'
    params_df2 = pd.read_csv(param_file_dir2)

    param_file_dir3 = directory + '/feature_dat.csv'
    params_df3 = pd.read_csv(param_file_dir3)

    param_file_dir4 = directory + '/inci_dat.csv'
    params_df4 = pd.read_csv(param_file_dir4)

    return params_df1.to_dict('records')[0], params_df2.to_dict('records')[0], params_df3, tf.convert_to_tensor(np.array(params_df4), dtype = tf.float32)

def make_synthetic_target(x, beta_score, pred_seqs):

    # The thredshold is the minimal beta score to be considered for selecting the candidate diets.
    ## This should be less full_seq_len - 1.
    thredshold = 20

    # Extract indices where the value is equal to 15 from beta_score vector and store the indices into candidate_idx
    candidates_idx = np.where(beta_score >= thredshold)[0]

    # Take the generated diets of extracted indices in case there exists at least one candidate index.
    if len(candidates_idx) > 0:

        # Define target buffer by copying the current real diets (input data)
        target_buffer = copy.deepcopy(np.array(x))

        # Replace the real diets of candidate indices (cadidate_idx) by generated diets of same indices.
        experiences = pred_seqs[candidates_idx, :]
        target_buffer[candidates_idx, :] = experiences

    # In case we have no candidate index, we just use target buffer defined by real diets.
    else:
        target_buffer = copy.deepcopy(np.array(x))

    return target_buffer

def update_dataset(epoch, BATCH_SIZE, target_buffer_update, target_buffer, tf_dataset_update, x, food_dict, nutrient_data):

    # Get indices of empty buffer if there are.
    empty_buffer_idx = [buffer_idx for buffer_idx, buffer_val in enumerate(target_buffer[0]) if len(buffer_val) == 0]

    # Update target_buffer every time when epoch + 1 is eqaul to the value of target_buffer_update.
    if (epoch + 1) % target_buffer_update == 0:

        ## transform target_buffer into the numpy type variable named eb_np, which is a short-cut of experience-buffer.
        ## note that target_buffer is a list of lists. 
        ## note that eb_np has a shape of (1, 10, 1072, 16) of which each element represents (num_of_batch, target_buffer_size, vocab_size, length_of_sequence).
        eb_np = np.array(target_buffer)
        
        ## make empty array with the same size of column (x.shape[1]).
        ## Here, x.shape[1] means the length of diet.
        new_diet_data_np = np.empty([0, x.shape[1]])

        ## 각 배치별로 target_buffer에서 하나의 beffuer_memory를 샘플링한 후 new_diet_data_np에 stack하기
        ## With regards to each batch, randomly sample a new batch to replace an existing batch from the target buffer, and construct a new dataset using the new batches.
        for batch in range(eb_np.shape[0]):

            # Fill empty buffer slot with the value in 0-th lost (if there are empty slots).
            if len(empty_buffer_idx) > 0:
                for idx in empty_buffer_idx:
                    eb_np[batch][idx] = eb_np[batch][0]

            # Calculate mean rewards of each slot in current buffer.
            ## Define empty array to save mean reward per buffer slot.
            mean_reward_per_slot = np.array([])

            ## Save per slot mean reward array and define weights for each buffer slot.
            for i in range(len(eb_np[batch])):
                rewards_per_slot = np.apply_along_axis(get_reward_ver2, axis = 1, arr = get_score_pandas(eb_np[batch][i], food_dict, nutrient_data), done = 0)[:, 0]
                mean_reward = np.array(rewards_per_slot).mean()
                mean_reward_per_slot = np.append(mean_reward_per_slot, mean_reward)

            # ## Calculate the weights along with the slot is sampled as a next target.
            # slot_weights = mean_reward_per_slot / np.sum(mean_reward_per_slot)
            # if (epoch + 1) % 90 == 0:
            #     print('slot_weights :', slot_weights)

            # selected_buffer_idx is the index to be randomly sampled at the given batch.
            # selected_buffer_idx = np.random.choice(eb_np[batch].shape[0])                         # 랜덤
            # selected_buffer_idx = np.random.choice(eb_np[batch].shape[0], p = slot_weights)       # 확률적
            selected_buffer_idx = np.argmax(mean_reward_per_slot)                                   # 그리디

            # define a new batch using the batch stored in selected_buffer_idx of eb_np.
            new_batch = eb_np[batch][selected_buffer_idx]

            # stack new batches row-wise and define it as new_diet_data_np which represents a new dataset.
            new_diet_data_np = np.append(new_diet_data_np, new_batch, axis = 0)

        ## Transform numpy object into tensor object for tensorflow to be worked, and do batch slicing.
        tf_dataset_update = tf.data.Dataset.from_tensor_slices(new_diet_data_np)
        tf_dataset_update = tf_dataset_update.batch(BATCH_SIZE, drop_remainder = True)

    return tf_dataset_update

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

        figure_dir = './figures/training_reward.png'
        plt.axhline(y = 9.29, color = "#d62728", linestyle = "-")
        plt.savefig(figure_dir, bbox_inches='tight')
    else:
        sns.lineplot(data = reward_df, x = 'epoch', y = 'reward', ci = 'sd')

def save_Gens(cp_dir, gen_seqs, food_dict):

    # Save the generated sequence and its sentence in 'generated_file_name' directory where a file name ends with the suffix '_sequence.csv' and with the suffix '_sentence.csv'. respectively.
    kwargs_vals = list(map(methodcaller("split", '='), cp_dir.split('/')[-2].split('--')))             # Get double-dash-splitted args and save in kwargs_vals
    kwargs_vals = np.array(kwargs_vals)[:, 1]
    kwargs_vals = np.append(np.array(cp_dir.split('/')[-4:-2]), kwargs_vals)                           # Append slash-splited args with kwargs_vals

    # Save generated sequences.
    # The sequence is a vector of tokens to which the number is allocated.
    generated_file_name = "/results/" + '--'.join(kwargs_vals) + "_sequence.csv"        
    pd.DataFrame(gen_seqs).to_csv(generated_file_name)
    
    # Save generated sentences.
    # The sentence is a vector that is composed of words corresponding to tokens.
    generated_file_name = "/results/" + '--'.join(kwargs_vals) + "_sentence.csv"
    pd.DataFrame(sequence_to_sentence(gen_seqs, food_dict)).to_csv(generated_file_name, encoding = 'utf-8-sig')

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
    figure_dir = './figures/tsne_mapping_plot.png'
    plt.savefig(figure_dir, bbox_inches='tight')

def rewards_matrix(epoch, rewards):

    rewards = np.array(rewards).reshape(-1, 1)

    epochs = np.repeat(epoch, len(rewards))
    epochs = epochs.reshape(-1, 1)

    samples = np.array(range(len(rewards)))
    samples = samples.reshape(-1, 1)

    per_epoch_rewards = pd.DataFrame(np.concatenate((epochs, rewards, samples), axis = 1))
    
    return per_epoch_rewards

def save_reward_df(reward_df, model, eb, bs, lr, num_epochs):
    dir_file_name = './results/' + model + '_eb=' + str(eb) + '_bs=' + str(bs) + '_lr=' + str(lr) + '_epoch=' + str(num_epochs) + '_rewards.csv'
    if not os.path.exists(dir_file_name):
        reward_df.to_csv(dir_file_name, index=False, mode='w')
    else:
        reward_df.to_csv(dir_file_name, index=False, mode='a', header=False)
    
    return dir_file_name

# def transition_matrix(diet_data_np, food_dict):
#     p_prob_mat = np.full([len(food_dict), len(food_dict)], fill_value = 0, dtype = np.float64)
#     for diet_idx in range(np.shape(diet_data_np)[0]):
#         for (i, j) in zip(diet_data_np[diet_idx, :].astype(int), diet_data_np[diet_idx, :][1:].astype(int)):
#             # print('i : {}, j : {}'.format(i, j))
#             p_prob_mat[i, j] += 1

#     p_prob_mat = p_prob_mat / p_prob_mat.sum(axis = 1, keepdims = True) 
#     p_prob_mat = np.nan_to_num(p_prob_mat, nan = 0.0, copy = True) + (1/len(food_dict))

#     return p_prob_mat

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

    print(' ')
    print('{} diets are deleted as they have more than {} empty slots'.format(len(delete_list), num_empty))

    # Get indices of diets that the number of empty is larger than num_empty.
    if empty_delete == True:
        non_value_idx = copy.deepcopy(delete_list)
                                                    

    # Delete indices of diets that the number of empty is larger than num_empty.
    diet_data_np = np.delete(diet_data_np, non_value_idx.astype(int), axis = 0) 

    return tf.cast(diet_data_np, dtype = tf.int32)

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
    if option == 'target':
        probas = np.random.choice(preds, size = 1, p = preds)
        action = np.where(preds == probas)[0][0] # 선택된 액션

    # greedy policy
    elif option == 'greedy':
        action = np.argmax(preds)
    
    # random policy
    elif option == 'random':
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

# get_score_pandas는 get_score_matrix로 reward를 계산하는데 오류를 발견하여 수정한 코드임
# 변경사항 (1) tensordot이 대신 pandas의 'name' 컬럼을 기준으로 list comprehension과 '.isin()' 함수를 활용해 nutrition을 가져오기.
def get_score_pandas(diet_batch, food_dict, nutrient_data):

    # nutrition 계산
    list_of_diets = sequence_to_sentence(diet_batch, food_dict)
    nutritions_of_diets = [nutrient_data[nutrient_data['name'].isin(each_diet)].sum(axis = 0)[1:] for each_diet in list_of_diets]
    # print('nutritions_of_diets :', nutritions_of_diets)

    # nutrient_vector로 변환
    nutrient_vector = np.array(nutritions_of_diets).reshape((diet_batch.shape[0], -1))

    return nutrient_vector
# def get_score_matrix(diet_batch, food_dict, nutrient_data):
#     # int 타입으로
#     diet_batch = tf.cast(diet_batch, dtype = tf.int64)

#     # onehot 벡터인데 time_step이 없는 one_hot으로
#     diet_batch = tf.reduce_sum(tf.one_hot(diet_batch, depth = len(food_dict)), axis = 1)

#     # 영양소 DB array 형태로
#     nutrient_array = np.array(nutrient_data.iloc[:, 1:])

#     # 각 식단별 영양소 계산
#     nutrient_vector = np.tensordot(diet_batch, nutrient_array, axes = 1)
#     return nutrient_vector

# 본 버전은 변경된 nutrition (final_new_nutrient.csv) 용 
# 변경사항 (1) weight 컬럼을 무시하기 위해 컬럼 index + 1씩 해줌 
# 변경사항 (2) 새 데이터 nutrition data.csv 에서 Retinol + beta-carotene 없이 바로 Vitamin A가 바로 존재하여, 이를 반영함.
# 변경사항 (3) 새 데이터 nutrition.csv 에서 Linolenic Acid와 Alpha-Linoleic Acid의 값이 mg으로 바뀐것을 고려하여, * 1e-3하여 g화 해주었음.
# 변경사항 (4) mode = 'add_breakfast' 제거
def get_reward_ver2(nutrient_state, done):
    '''
    영양보상 계산
    done : determine whether to terminate the function
    '''
    # score_vector로부터 reward 뽑는 함수
    # 행위로부터 결정된 보상을 제공하는 함수

    nutrient_reward = 0
    nutrient_reward_set = np.zeros([15])

    total_calorie = nutrient_state[1]         # 'Energy'
    total_c = nutrient_state[4] * 4           # 'Carbohydrate' (kcal)
    total_p = nutrient_state[2] * 4           # 'Protein' (kcal)
    total_p_gram = nutrient_state[2]          # 'Protein' (g)
    total_f = nutrient_state[3] * 9           # 'Fat' (kcal)
    total_dietary = nutrient_state[5]         # 'Total Dietary Fiber' (g)
    total_calcium = nutrient_state[6]         # 'Calcium' (mg)
    total_iron = nutrient_state[7]            # 'Iron' (mg)
    total_sodium = nutrient_state[8]          # 'Sodium' (mg)
    total_vitaA = nutrient_state[9]           # 'Vitamin A' (mg RAE)
    total_vitaB1 = nutrient_state[10]         # 'Vitamin B1 (Thiamine)' (mg)
    total_vitaB2 = nutrient_state[11]         # 'Vitamin B2 (Rivoflavin)' (mg)
    total_vitaC = nutrient_state[12]          # 'Vitamin C' (mg)
    total_lino = nutrient_state[13] * 1e-3    # 'Linoleic Acid' (g)
    total_alpha = nutrient_state[14] * 1e-3   # 'Alpha-Linoleic Acid' (g)


    # 영양보상 1. 총 열량 (kcal)
    if total_calorie >= 1260 and total_calorie <= 1540:
        nutrient_reward += 1
        nutrient_reward_set[0] +=1

    # 영양보상 2. 단백질 (g)
    if total_p_gram >= 20:
        nutrient_reward += 1
        nutrient_reward_set[1] +=1

    # 영양보상 3. 총 식이섬유 (g)
    if total_dietary >= 11 and total_dietary <= 20:
        nutrient_reward += 1
        nutrient_reward_set[2] +=1

    # 영양보상 4. 비타민 A (mugRAE)
    if total_vitaA >= 230 and total_vitaA <= 750:
        nutrient_reward += 1
        nutrient_reward_set[3] +=1

    # 영양보상 5. 비타민 C (mg)
    if total_vitaC >= 35 and total_vitaC <= 510:
        nutrient_reward += 1
        nutrient_reward_set[4] +=1

    # 영양보상 6. 비타민 B1 (mg)
    if total_vitaB1 >= 0.4:
        nutrient_reward += 1
        nutrient_reward_set[5] +=1

    # 영양보상 7. 비타민 B2 (mg)
    if total_vitaB2 >= 0.5:
        nutrient_reward += 1
        nutrient_reward_set[6] +=1

    # 영양보상 8. 칼슘 (mg) 
    if total_calcium >= 500 and total_calcium <= 2500:
        nutrient_reward += 1
        nutrient_reward_set[7] +=1

    # 영양보상 9. 철 (mg)
    if total_iron >= 5 and total_iron <= 40:
        nutrient_reward += 1
        nutrient_reward_set[8] +=1

    # 영양보상 10. 나트륨 (mg)
    if total_sodium <= 1600:
        nutrient_reward += 1
        nutrient_reward_set[9] +=1

    # 영양보상 11. 리놀레산 (g)
    if total_lino >= 4.6 and total_lino <= 9.1:
        nutrient_reward += 1
        nutrient_reward_set[10] +=1

    # 영양보상 12. 알파 리놀렌산 (g)
    if total_alpha >= 0.6 and total_alpha <= 1.17:
        nutrient_reward += 1
        nutrient_reward_set[11] +=1

    # 영양보상 13. 탄수화물 비율 (kcal %)
    if total_c >= total_calorie * 0.55 and total_c <= total_calorie * 0.65:
        nutrient_reward += 1
        nutrient_reward_set[12] +=1

    # 영양보상 14. 단백질 비율 (kcal %)
    if total_p >= total_calorie * 0.07 and total_p <= total_calorie * 0.2:
        nutrient_reward += 1
        nutrient_reward_set[13] +=1

    # 영양보상 15. 지방 비율 (kcal %)
    if total_f >= total_calorie * 0.15 and total_f <= total_calorie * 0.3:
        nutrient_reward += 1
        nutrient_reward_set[14] +=1

    # return nutrient_reward, done, nutrient_reward_set, composition_reward
    return nutrient_reward, done, nutrient_reward_set


# def get_reward_ver2(nutrient_state, done, mode):
#     '''
#     영양보상 계산
#     done : determine whether to terminate the function
#     mode : determine whether to apply reward standard of diet with or without breakfast.
#     '''
#     # score_vector로부터 reward 뽑는 함수
#     # 행위로부터 결정된 보상을 제공하는 함수

#     nutrient_reward = 0
#     nutrient_reward_set = np.zeros([15])

#     total_calorie = nutrient_state[0] # 'Energy'
#     total_c = nutrient_state[3] * 4 # 'Carbohydrate' (kcal)
#     total_p = nutrient_state[1] * 4 # 'Protein' (kcal)
#     total_p_gram = nutrient_state[1] # 'Protein' (g)
#     total_f = nutrient_state[2] * 9 # 'Fat' (kcal)
#     total_dietary = nutrient_state[4] # 'Total Dietary Fiber' (g)
#     total_calcium = nutrient_state[5] # 'Calcium' (mg)
#     total_iron = nutrient_state[6] # 'Iron' (mg)
#     total_sodium = nutrient_state[7] # 'Sodium' (mg)
#     total_vitaA = nutrient_state[8] + nutrient_state[9] # 'Vitamin A' = Retinol + beta-carotene (mugRAE)
#     total_vitaB1 = nutrient_state[10]   # 'Vitamin B1 (Thiamine)' (mg)
#     total_vitaB2 = nutrient_state[11]   # 'Vitamin B2 (Rivoflavin)' (mg)
#     total_vitaC = nutrient_state[12]    # 'Vitamin C' (mg)
#     total_lino = nutrient_state[13]     # 'Linoleic Acid' (g)
#     total_alpha = nutrient_state[14]    # 'Alpha-Linoleic Acid' (g)

#     # If you're using diet data with breakfast (i.e., kwargs['add_breakfast'] == True)
#     if mode == True:

#         # 영양보상 1. 총 열량 (kcal)
#         if total_calorie >= 1260 and total_calorie <= 1540:
#             nutrient_reward += 1
#             nutrient_reward_set[0] +=1

#         # 영양보상 2. 단백질 (g)
#         if total_p_gram >= 20:
#             nutrient_reward += 1
#             nutrient_reward_set[1] +=1

#         # 영양보상 3. 총 식이섬유 (g)
#         if total_dietary >= 11 and total_dietary <= 20:
#             nutrient_reward += 1
#             nutrient_reward_set[2] +=1

#         # 영양보상 4. 비타민 A (mugRAE)
#         if total_vitaA >= 230 and total_vitaA <= 750:
#             nutrient_reward += 1
#             nutrient_reward_set[3] +=1

#         # 영양보상 5. 비타민 C (mg)
#         if total_vitaC >= 35 and total_vitaC <= 510:
#             nutrient_reward += 1
#             nutrient_reward_set[4] +=1

#         # 영양보상 6. 비타민 B1 (mg)
#         if total_vitaB1 >= 0.4:
#             nutrient_reward += 1
#             nutrient_reward_set[5] +=1

#         # 영양보상 7. 비타민 B2 (mg)
#         if total_vitaB2 >= 0.5:
#             nutrient_reward += 1
#             nutrient_reward_set[6] +=1

#         # 영양보상 8. 칼슘 (mg) 
#         if total_calcium >= 500 and total_calcium <= 2500:
#             nutrient_reward += 1
#             nutrient_reward_set[7] +=1

#         # 영양보상 9. 철 (mg)
#         if total_iron >= 5 and total_iron <= 40:
#             nutrient_reward += 1
#             nutrient_reward_set[8] +=1

#         # 영양보상 10. 나트륨 (mg)
#         if total_sodium <= 1600:
#             nutrient_reward += 1
#             nutrient_reward_set[9] +=1

#         # 영양보상 11. 리놀레산 (g)
#         if total_lino >= 4.6 and total_lino <= 9.1:
#             nutrient_reward += 1
#             nutrient_reward_set[10] +=1

#         # 영양보상 12. 알파 리놀렌산 (g)
#         if total_alpha >= 0.6 and total_alpha <= 1.17:
#             nutrient_reward += 1
#             nutrient_reward_set[11] +=1

#         # # 영양보상 13. 탄단지 비율 (kcal %)
#         # if ((total_c >= total_calorie * 0.55 and total_c <= total_calorie * 0.65) and 
#         # (total_p >= total_calorie * 0.07 and total_p <= total_calorie * 0.2) and 
#         # (total_f >= total_calorie * 0.15 and total_f <= total_calorie * 0.3)):
#         #     nutrient_reward += 1
#         #     nutrient_reward_set[12] +=1        

#         # 영양보상 13. 탄수화물 비율 (kcal %)
#         if total_c >= total_calorie * 0.55 and total_c <= total_calorie * 0.65:
#             nutrient_reward += 1
#             nutrient_reward_set[12] +=1

#         # 영양보상 14. 단백질 비율 (kcal %)
#         if total_p >= total_calorie * 0.07 and total_p <= total_calorie * 0.2:
#             nutrient_reward += 1
#             nutrient_reward_set[13] +=1

#         # 영양보상 15. 지방 비율 (kcal %)
#         if total_f >= total_calorie * 0.15 and total_f <= total_calorie * 0.3:
#             nutrient_reward += 1
#             nutrient_reward_set[14] +=1

#     # If you're using diet data without breakfast (i.e., kwargs['add_breakfast'] == False)
#     elif mode == False:

#         # 영양보상 1. 총 열량 (kcal)
#         if total_calorie >= 945 and total_calorie <= 1155:
#             nutrient_reward += 1
#             nutrient_reward_set[0] +=1

#         # 영양보상 2. 단백질 (g)
#         if total_p_gram >= 15:
#             nutrient_reward += 1
#             nutrient_reward_set[1] +=1

#         # 영양보상 3. 총 식이섬유 (g)
#         if total_dietary >= 8.25 and total_dietary <= 15:
#             nutrient_reward += 1
#             nutrient_reward_set[2] +=1

#         # 영양보상 4. 비타민 A (mugRAE)
#         if total_vitaA >= 172.5 and total_vitaA <= 562.5:
#             nutrient_reward += 1
#             nutrient_reward_set[3] +=1

#         # 영양보상 5. 비타민 C (mg)
#         if total_vitaC >= 26.25 and total_vitaC <= 382.5:
#             nutrient_reward += 1
#             nutrient_reward_set[4] +=1

#         # 영양보상 6. 비타민 B1 (mg)
#         if total_vitaB1 >= 0.3:
#             nutrient_reward += 1
#             nutrient_reward_set[5] +=1

#         # 영양보상 7. 비타민 B2 (mg)
#         if total_vitaB2 >= 0.375:
#             nutrient_reward += 1
#             nutrient_reward_set[6] +=1

#         # 영양보상 8. 칼슘 (mg) 
#         if total_calcium >= 375 and total_calcium <= 1875:
#             nutrient_reward += 1
#             nutrient_reward_set[7] +=1

#         # 영양보상 9. 철 (mg)
#         if total_iron >= 3.75 and total_iron <= 30:
#             nutrient_reward += 1
#             nutrient_reward_set[8] +=1

#         # 영양보상 10. 나트륨 (mg)
#         if total_sodium <= 1200:
#             nutrient_reward += 1
#             nutrient_reward_set[9] +=1

#         # 영양보상 11. 리놀레산 (g)
#         if total_lino >= 3.3 and total_lino <= 6.8:
#             nutrient_reward += 1
#             nutrient_reward_set[10] +=1

#         # 영양보상 12. 알파 리놀렌산 (g)
#         if total_alpha >= 0.4 and total_alpha <= 0.9:
#             nutrient_reward += 1
#             nutrient_reward_set[11] +=1

#         # # 영양보상 13. 탄단지 비율 (kcal %)
#         # if ((total_c >= total_calorie * 0.55 and total_c <= total_calorie * 0.65) and 
#         # (total_p >= total_calorie * 0.07 and total_p <= total_calorie * 0.2) and 
#         # (total_f >= total_calorie * 0.15 and total_f <= total_calorie * 0.3)):
#         #     nutrient_reward += 1
#         #     nutrient_reward_set[12] +=1

#         # 영양보상 13. 탄수화물 비율 (kcal %)
#         if total_c >= total_calorie * 0.55 and total_c <= total_calorie * 0.65:
#             nutrient_reward += 1
#             nutrient_reward_set[12] +=1

#         # 영양보상 14. 단백질 비율 (kcal %)
#         if total_p >= total_calorie * 0.07 and total_p <= total_calorie * 0.2:
#             nutrient_reward += 1
#             nutrient_reward_set[13] +=1

#         # 영양보상 15. 지방 비율 (kcal %)
#         if total_f >= total_calorie * 0.15 and total_f <= total_calorie * 0.3:
#             nutrient_reward += 1
#             nutrient_reward_set[14] +=1


#     # return nutrient_reward, done, nutrient_reward_set, composition_reward
#     return nutrient_reward, done, nutrient_reward_set

def get_score_vector(diet, nutrient_data):
    '''
    diet는 array 타입의 tokenize된 식단 벡터이어야 함
    '''
    # 영양점수
    target_foods_mat = nutrient_data.iloc[diet, 1:]
    nutrient_vector = target_foods_mat.sum(axis = 0)

    return nutrient_vector

from matplotlib import pyplot as plt
def plot_reward_dist(reward_vec):
    reward = np.unique(reward_vec, return_counts = True)[0]
    freq = np.unique(reward_vec, return_counts = True)[1]
    plt.bar(reward, freq)

# 한영번역기
translate_book = pd.read_csv('/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Data (new)/translated_menu_name (수정).csv', encoding='CP949')
translate_dict = dict(list(zip(translate_book['kor'], translate_book['eng'])))
def kor_to_eng(sentence_list, refer = translate_dict):

    gen_food_list = []

    for i, val in enumerate(sentence_list):
        each_food = []

        for j, val2 in enumerate(val):
            each_food.append(translate_dict[val2])

        gen_food_list.append(each_food)

    return(gen_food_list)

def sentence_trim(any_sentence, cut_len):
    trim_sentence = [(x[:cut_len] + '...') if len(str(x)) > cut_len else x for x in any_sentence[0]]
    return trim_sentence

def plot_attention(sample_num, attention, sentence, predicted_sentence, food_dict, reference, language = 'kor'):
    if len(sentence.shape) == 1:
        sentence = sequence_to_sentence([sentence], food_dict)
        predicted_sentence = sequence_to_sentence([predicted_sentence], food_dict)

    if language == 'eng':
        sentence = kor_to_eng(sentence, refer = reference)
        predicted_sentence = kor_to_eng(predicted_sentence, refer = reference)

    # ...으로 요약해서 보여주기
    sentence = sentence_trim(sentence, cut_len = 8)
    predicted_sentence = sentence_trim(predicted_sentence, cut_len = 8)

    sentence = np.squeeze(sentence)
    predicted_sentence = np.squeeze(predicted_sentence)

    font1 = {'family':'NanumMyeongjo',
             'size' : 17.5}

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    attention = attention[:len(predicted_sentence)-2, :len(sentence)-2]
    ax.matshow(attention, cmap='viridis', vmin=0.0)

    ax.set_xticklabels(sentence[:-1], rotation=90, fontdict=font1)
    ax.set_yticklabels(predicted_sentence[:-1], fontdict=font1)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, len(predicted_sentence)-2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(sentence)-3, 1), minor=True)

    ax.grid(which = 'minor', color='w', linestyle='-', linewidth=2)

    ax.set_xlabel('Input text')
    ax.set_ylabel('Output text')
    # plt.suptitle('Attention weights')
    plt.savefig('./figures/' + str(language) + '_attention_map(' + str(sample_num) +').png', dpi=300)

from scipy.spatial import distance
def get_most_change_sequence(real_seqs, gen_seqs, top_n):
    dist_by_pair = []  # dimension : (batch_size, )
    for i in range(gen_seqs.shape[0]):
        dist_by_pair = np.append(dist_by_pair, distance.jaccard(gen_seqs[i], real_seqs.numpy()[i]))

    dist_index = np.arange(0, len(dist_by_pair)).reshape((-1, 1))
    dist_by_pair = np.around(dist_by_pair.reshape(-1, 1), decimals = 3)
 
    dist_by_index_and_pair = np.concatenate((dist_index, dist_by_pair), axis = 1)   # dimension : (500, 2)

    dist_top_rank = dist_by_index_and_pair[dist_by_index_and_pair[:, 1].argsort(), :][-top_n:]
    result = pd.DataFrame(dist_top_rank)
    result.columns = ["example_idx", "jaccard_dist"]
    result = result.astype({'example_idx' : 'int'})

    return result

def ingredient_tokenization(diet_ingred_sequences, ingred_dict):
    diet_ingred_tokenize = []
    for diet_idx, diet in enumerate(diet_ingred_sequences):
        diet_ingred_tokenize.append([ingred_dict[ingred] for ingred in diet])

    return diet_ingred_tokenize


# This function transforms a given diet sequence of simple string type, into an object that can be referred by any embedded classes and functions provided by dietkit package.
def objectification(sample_diet_sequences, menus_dict):
    ## process string error.
    ### (1) make all menus as lowercase 
    ### (2) remove blank in diet
    ### (3) make first letter as capital letter
    ### (4) but menu 'empty' is exception.

    for diet_idx, diet in enumerate(sample_diet_sequences):
        for menu_idx, menu in enumerate(diet): 
            if diet[menu_idx] != "empty":
                diet[menu_idx] = menu.lower().strip().capitalize()
        sample_diet_sequences[diet_idx] = diet

    ## change diet as object_diet which is an object-type of diet addressed by dietkit, otherwise you cannot use dietkit.
    object_diet_sequences = []
    for diet_idx, diet in enumerate(sample_diet_sequences):
        object_diet_sequences.append([ menus_dict[menu] for menu in diet ])

    object_diet_sequences = Diet(dict(list(zip(range(len(object_diet_sequences)), object_diet_sequences[:]))))
    
    return object_diet_sequences