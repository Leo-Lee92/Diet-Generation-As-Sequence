# %%
import tensorflow as tf
import numpy as np
import pandas as pd
import copy

# # 모듈로 돌릴 때 필요한 것 패키지
# from model import *
# from preprocessing import *
# from util import *

# (1) Define Teacher
Teacher2 = RNN()
inputs = np.ones([BATCH_SIZE, 1]) * [1726]
h = Teacher2.initialize_hidden_state(BATCH_SIZE)
Teacher2(inputs, h)
Teacher2.load_weights('/home/messy92/Leo/Project_Gosin/SL+RL/Food_Generation/pretrain_folder/teacher_weights.tf')
# Teacher2.set_weights(Teacher.get_weights())

# (2) Define Actor
Actor = None
actor_mode = None
actor_mode = input('Mode of Actor : ')
if actor_mode == "RNN":
    '''
    RNN 구조를 Actor에 활용
    '''
    Actor = RNN()
    print("RNN Actor Selected !")
elif actor_mode == "MLP":
    '''
    MLP 구조를 Actor에 활용
    '''
    Actor = RLActor()
    print("MLP Actor Selected !")

# (3) Define Critic
Critic = None
Critic = RLCritic()

# (4) Define Hyper-parameters
double_opac = "None"
closed_loop = "None"
lambda_param = "None"
update_epoch = None
update_method = "None"
likelihood_sampling = "NONE"
Target_Actor = None

likelihood_sampling = input('likeilhood_sampling ? : LS or No')
# (5) Define Target_Actor
double_opac = input("Double OPAC ? : Yes or No")
if double_opac == "Yes":
    closed_loop = input('Closed-Loop Learning? : CL or OL')
    lambda_param = float(input('How much OPAC will be consider Closed-Loop (ex., 1e-5)'))
    update_epoch = int(input('How frequent will you update OPAC from Closed-Loop (ex., 5)'))
    update_method = input('Select update method : smooth vs switch')
    if actor_mode == "RNN":
        Target_Actor = RNN()
    elif actor_mode == "MLP":
        Target_Actor = RLActor()


'''
other parameters
'''
seqlen = 16
optimizer = tf.keras.optimizers.Adam(1e-3)

per_epoch_mean_t_complete_score_vector_list = []
per_epoch_mean_a_complete_score_vector_list = []
per_epoch_mean_ta_complete_score_vector_list = []
per_epoch_mean_prob_score_a_list = []
per_epoch_mean_prob_score_ta_list = []

mean_t_complete_score_vector_list_all = np.empty((0, 5))
mean_a_complete_score_vector_list_all = np.empty((0, 5))
mean_ta_complete_score_vector_list_all = np.empty((0, 5))
mean_a_prob_score_vector_list_all = np.empty((0, 5))
mean_ta_prob_score_vector_list_all = np.empty((0, 5))

ta_nutrient_state = 0
prob_score_ta = 0


num_epochs = 300
for epochs in range(num_epochs):
# Keep_On = True
# while(Keep_On):
    actor_batch_loss = 0
    critic_batch_loss = 0

    mean_t_complete_score_vector_list = np.array([])
    mean_a_complete_score_vector_list = np.array([])
    mean_ta_complete_score_vector_list = np.array([])


    mean_prob_score_a_list = np.array([])
    mean_prob_score_ta_list = np.array([])

    for x in tf_dataset:

        inputs = np.ones([BATCH_SIZE, 1]) * [1726]

        t_hidden_state = Teacher2.initialize_hidden_state(BATCH_SIZE)
        t_gen_seq = copy.deepcopy(inputs)
        a_gen_seq = copy.deepcopy(inputs)
        ta_gen_seq = copy.deepcopy(inputs)
        online_gen_seq = copy.deepcopy(inputs)


        a_pre_score_vector = tf.convert_to_tensor(np.zeros([BATCH_SIZE]), dtype = tf.float32)

        actor_loss_sum = 0
        critic_loss_sum = 0
        done = 0

        batch_prob_score_a = 0
        batch_prob_score_ta = 0

        '''
        [시작] 토큰 다음 첫번째 토큰을 샘플링
        '''
        # Behavioral Policy로서 Teacher 모델 활용
        t_outputs, t_hidden_state = Teacher2(inputs, t_hidden_state)
        result = np.apply_along_axis(get_action, axis = 1, arr = t_outputs, option = "prob")
        t_actions = tf.cast(tf.expand_dims(result[:, 0], 1), tf.int32)

        # [시작] 이후 첫번째 토큰을 stack해주기
        inputs = copy.deepcopy(t_actions)              # inputs변수를 다시 선언
        a_actions = copy.deepcopy(t_actions)
        ta_actions = copy.deepcopy(t_actions)
        online_actions = copy.deepcopy(t_actions)

        t_gen_seq = np.hstack([t_gen_seq, inputs])
        a_gen_seq = np.hstack([a_gen_seq, a_actions])
        ta_gen_seq = np.hstack([ta_gen_seq, ta_actions])
        online_gen_seq = np.hstack([online_gen_seq, online_actions])

        a_hidden_state = Actor.initialize_hidden_state(BATCH_SIZE)          # 각 모델의 hidden_state를 Teacher의 최초 hidden state로 선언
        online_hidden_state = Actor.initialize_hidden_state(BATCH_SIZE)

        a_pre_nutrient_state = np.zeros([32, 21])

        if Target_Actor != None:
            ta_hidden_state = Target_Actor.initialize_hidden_state(BATCH_SIZE)
            ta_pre_score_vector = tf.convert_to_tensor(np.zeros([BATCH_SIZE]), dtype = tf.float32)

        for i in range(1, seqlen - 2):

            '''
            실시간 (online) 생성 결과 확인하기
            '''
            if actor_mode == "RNN":
                online_outputs, online_hidden_state_next = Actor(online_actions, online_hidden_state)
            else:
                online_outputs = Actor(online_actions)

            online_result = np.apply_along_axis(get_action, axis = 1, arr = online_outputs, option = "prob")
            online_actions = tf.cast(tf.expand_dims(online_result[:, 0], 1), tf.int32)
            online_gen_seq = np.hstack([online_gen_seq, online_actions])
            '''
            실시간 (online) 생성 결과 확인하기
            '''


            '''
            Teacher
            '''
            # Behavioral Policy로서 Teacher 모델 활용
            t_outputs, t_hidden_state_next = Teacher2(inputs, t_hidden_state)
            t_result = np.apply_along_axis(get_action, axis = 1, arr = t_outputs, option = "prob")
            t_actions = tf.cast(tf.expand_dims(t_result[:, 0], 1), tf.int32)

            '''
            Samples
            '''
            # t_sample = {current_state, step_up, next_state}
            ## current_state : (inputs, t_hidden_state)
            ## step_up : (t_outputs, t_actions)
            ## next_state : (t_actions, t_hidden_state_next)
            t_sample = ((inputs, t_hidden_state, t_outputs, t_actions, t_hidden_state_next))

            '''
            OPAC Training
            '''
            # Not Double-OPAC
            if double_opac == "No":
                if actor_mode == "RNN":
                    a_actions, a_hidden_state_next, prob_score_a, a_score_vector, t_actions, t_gen_seq, a_nutrient_state = Training_OPAC(Actor, Critic, None, likelihood_sampling, t_gen_seq, None, t_sample, None, a_hidden_state, None, a_pre_score_vector, None, a_pre_nutrient_state, ta_pre_nutrient_state, food_dict, actor_mode)
                    a_hidden_state = copy.deepcopy(a_hidden_state_next)

                else:
                    a_actions, prob_score_a, a_score_vector, t_actions, t_gen_seq, a_nutrient_state = Training_OPAC(Actor, Critic, None, likelihood_sampling, t_gen_seq, None, t_sample, None, None, None, a_pre_score_vector, None, a_pre_nutrient_state, a_pre_nutrient_state, food_dict, actor_mode)

            # Double-OPAC
            else:
                # Training_OPAC()의 인자들 중 a_gen_seq <- ta_gen_seq; a_actions <- ta_actions 로 바꾸면 Professor-Forcing이 됨.

                if closed_loop == "OL":
                    pred_trajectory = copy.deepcopy(a_gen_seq)
                    pred_sample = copy.deepcopy(a_actions)
                else:
                    pred_trajectory = copy.deepcopy(ta_gen_seq)
                    pred_sample = copy.deepcopy(ta_actions)

                if actor_mode == "RNN":
                    a_actions, a_hidden_state_next, prob_score_a, prob_score_ta, a_score_vector, ta_score_vector, t_actions, t_gen_seq, ta_actions, ta_hidden_state_next, a_nutrient_state, ta_nutrient_state = Training_OPAC(Actor, Critic, Target_Actor, likelihood_sampling, t_gen_seq, pred_trajectory, t_sample, pred_sample, a_hidden_state, ta_hidden_state, a_pre_score_vector, ta_pre_score_vector, a_pre_nutrient_state, a_pre_nutrient_state, food_dict, actor_mode)
                    ta_hidden_state = copy.deepcopy(ta_hidden_state_next)
                    a_hidden_state = copy.deepcopy(a_hidden_state_next)

                else:
                    a_actions, prob_score_a, prob_score_ta, a_score_vector, ta_score_vector, t_actions, t_gen_seq, ta_actions, a_nutrient_state, ta_nutrient_state = Training_OPAC(Actor, Critic, Target_Actor, likelihood_sampling, t_gen_seq, pred_trajectory, t_sample, pred_sample, None, None, a_pre_score_vector, ta_pre_score_vector, a_pre_nutrient_state, a_pre_nutrient_state, food_dict, actor_mode)

                ta_gen_seq = np.hstack([ta_gen_seq, ta_actions])         # 중간생성결과를 보기 위함
                ta_pre_score_vector = copy.deepcopy(ta_score_vector)

            '''
            Prediction or Generating : stakcing foods according to target policy
            '''
            a_gen_seq = np.hstack([a_gen_seq, a_actions])       # 중간생성결과를 보기 위함 + double-OPAC에서 trajectory

            '''
            Step up
            '''
            inputs = copy.deepcopy(t_actions)                       # 선택한 action으로 다음 state 정의
            t_hidden_state = copy.deepcopy(t_hidden_state_next)     # 반환된 hidden_state로 다음 state 정의
            a_pre_nutrient_state = copy.deepcopy(a_nutrient_state)
            
            if i != 14:         # 이전 reward 정의
                a_pre_score_vector = copy.deepcopy(a_score_vector)

            # (1) 확률추적
            batch_prob_score_a += prob_score_a
            batch_prob_score_ta += prob_score_ta

        '''
        최종 영양점수 보기
        '''
        mean_t_complete_score_vector = return_nutrient_score(t_gen_seq, nutrient_data)
        mean_a_complete_score_vector = return_nutrient_score(a_gen_seq, nutrient_data)
        mean_ta_complete_score_vector = return_nutrient_score(ta_gen_seq, nutrient_data)


        mean_t_complete_score_vector_list = np.append(mean_t_complete_score_vector_list, mean_t_complete_score_vector)
        mean_a_complete_score_vector_list = np.append(mean_a_complete_score_vector_list, mean_a_complete_score_vector)
        mean_ta_complete_score_vector_list = np.append(mean_ta_complete_score_vector_list, mean_ta_complete_score_vector)

        '''
        최종 평균 우도점수 보기
        '''
        batch_prob_score_a = batch_prob_score_a / (seqlen - 1)
        mean_prob_score_a = tf.reduce_mean(batch_prob_score_a)
        mean_prob_score_a_list = np.append(mean_prob_score_a_list, mean_prob_score_a)

        batch_prob_score_ta = batch_prob_score_ta / (seqlen - 1)
        mean_prob_score_ta = tf.reduce_mean(batch_prob_score_ta)
        mean_prob_score_ta_list = np.append(mean_prob_score_ta_list, mean_prob_score_ta)

        '''
        Plot 그림용 데이터 저장하는 함수
        '''
        # Reward
        mean_t_complete_score_vector_list_all = summarize_episode(epochs, mean_t_complete_score_vector, "T", "RNN", "None", mean_t_complete_score_vector_list_all)
        mean_a_complete_score_vector_list_all = summarize_episode(epochs, mean_a_complete_score_vector, "A", actor_mode, likelihood_sampling + "-" + closed_loop + "(" + update_method + ")", mean_a_complete_score_vector_list_all)
        mean_ta_complete_score_vector_list_tall = summarize_episode(epochs, mean_ta_complete_score_vector, "A", actor_mode, likelihood_sampling + "-" + closed_loop + "(" + update_method + ")", mean_ta_complete_score_vector_list_all)

        # mean_t_complete_score_vector_list_all = summarize_episode_v1(epochs, mean_t_complete_score_vector, mean_t_complete_score_vector_list_all, "T", actor_mode + " with LS:" + likelihood_sampling + " with FCL:" + fixed_cross_likelihood + " with Double:" + double_opac, None, "OPAC")
        # mean_a_complete_score_vector_list_all = summarize_episode_v1(epochs, mean_a_complete_score_vector, mean_a_complete_score_vector_list_all, "A", actor_mode + " with LS:" + likelihood_sampling + " with FCL:" + fixed_cross_likelihood + " with Double:" + double_opac, None,  "OPAC")

        # Accuracy
        mean_a_prob_score_vector_list_all = summarize_episode(epochs, mean_prob_score_a, "A", actor_mode, likelihood_sampling + "-" + closed_loop + "(" + update_method + ")", mean_a_prob_score_vector_list_all)
        mean_ta_prob_score_vector_list_tall = summarize_episode(epochs, mean_prob_score_ta, "A", actor_mode, likelihood_sampling + "-" + closed_loop + "(" + update_method + ")", mean_ta_prob_score_vector_list_all)
        # mean_a_prob_score_vector_list_all = summarize_episode_v1(epochs, mean_prob_score_a, mean_a_prob_score_vector_list_all, "A", actor_mode + " with LS:" + likelihood_sampling + " with FCL:" + fixed_cross_likelihood + " with Double:" + double_opac, None, "OPAC")
        
        '''
        Loss 계산
        '''
        actor_loss_sum = actor_loss_sum / seqlen
        critic_loss_sum = critic_loss_sum / seqlen

        actor_batch_loss += actor_loss_sum
        critic_batch_loss += critic_loss_sum

    if epochs % 5 == 0:
        print(' ')
        print('Teacher')
        print(sequence_to_sentence(t_gen_seq[:, 1:15], food_dict)[0])
        print(' ')
        print('OPAC')
        print(sequence_to_sentence(a_gen_seq[:, 1:15], food_dict)[0])
        print('actor_mode : {}'.format(actor_mode))
        print('likelihood_sampling : {}'.format(likelihood_sampling))

        print(' ')
        print('Online')
        print(sequence_to_sentence(online_gen_seq[:, 1:15], food_dict)[0])

    if double_opac == "Yes":
        if epochs % 5 == 0:
            print(' ')
            print('Target OPAC')
            print(sequence_to_sentence(ta_gen_seq[:, 1:15], food_dict)[0])
            print('double-opac : {}'.format(double_opac))
            print('update_epochs : {}'.format(update_epoch))
            print('lambda_param : {}'.format(lambda_param))

        # '''
        # Update Actor Every 'update_epoch' Epochs
        # '''
        # if epochs % update_epoch == 0:

        #     '''
        #     Smoothing Update
        #     '''
        #     if update_method != "None":

        #         if update_method == "smooth":
        #             # new_weight = calculate_new_weights(Actor.get_weights(), Target_Actor.get_weights(), lambda_param)   # Calculate New Weights for Target_Actor
        #             new_weight = calculate_new_weights(Target_Actor.get_weights(), Actor.get_weights(), lambda_param)   # Calculate New Weights for Target_Actor
        #             # Actor.set_weights(new_weight)   # Update Actor
        #             Target_Actor.set_weights(new_weight)   # Update Target_Actor

        #         '''
        #         Switching Update
        #         '''
        #         if update_method == "switch":
        #             new_weight1 = calculate_new_weights(Actor.get_weights(), Target_Actor.get_weights(), 1)   # Take 100% of Target_Actor's Weights and save it to new_weight1
        #             new_weight2 = calculate_new_weights(Actor.get_weights(), Target_Actor.get_weights(), 0)   # Take 100% of Actor's Weights and save it to new_weight2

        #             Actor.set_weights(new_weight1)          # Update Actor with Target_Actor's weights
        #             Target_Actor.set_weights(new_weight2)   # Update Target_Actor with Actor's weights


    per_epoch_mean_t_complete_score_vector_list.append(tf.reduce_mean(mean_t_complete_score_vector_list))
    per_epoch_mean_a_complete_score_vector_list.append(tf.reduce_mean(mean_a_complete_score_vector_list))
    per_epoch_mean_ta_complete_score_vector_list.append(tf.reduce_mean(mean_ta_complete_score_vector_list))

    per_epoch_mean_prob_score_a_list.append(tf.reduce_mean(mean_prob_score_a_list))
    per_epoch_mean_prob_score_ta_list.append(tf.reduce_mean(mean_prob_score_ta_list))

    total_actor_loss = actor_batch_loss / len(list(tf_dataset))
    total_critic_loss = critic_batch_loss / len(list(tf_dataset))

    print(' ')
    print('epoch is {}, total actor loss is : {}, and total critic loss is :{}'.format(epochs, total_actor_loss, total_critic_loss))
    print('epoch is {}, real_nutrient_score is : {}'.format(epochs, per_epoch_mean_t_complete_score_vector_list[-1]))
    print('epoch is {}, actor_nutrient_score is : {}'.format(epochs, per_epoch_mean_a_complete_score_vector_list[-1]))
    print('epoch is {}, actor_prob_score is : {}'.format(epochs, per_epoch_mean_prob_score_a_list[-1]))
    print('epoch is {}, target_actor_nutrient_score is : {}'.format(epochs, per_epoch_mean_ta_complete_score_vector_list[-1]))
    print('epoch is {}, target_actor_prob_score is : {}'.format(epochs, per_epoch_mean_prob_score_ta_list[-1]))

    epochs += 1

# %%
# 보상 플로팅
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from matplotlib import rc
# target_dat = np.vstack([mean_t_complete_score_vector_list_all, mean_a_complete_score_vector_list_all])
real_mean_nutrient_score = return_nutrient_score(diet_data_np, nutrient_data)

target_dat = mean_a_complete_score_vector_list_all
reward_df = pd.DataFrame(target_dat)
reward_df.columns = ["Epoch", "Mean Reward", "Agent", "Network", "condition"]

reward_df['Epoch'] = pd.to_numeric(reward_df['Epoch'], errors = "coerce").astype(int)
reward_df['Mean Reward'] = pd.to_numeric(reward_df['Mean Reward'], errors = "coerce")
save_result_name1 = "LS:" + likelihood_sampling + "-CL:" + closed_loop + "-Network:" + actor_mode + "-measure1.csv"
reward_df.to_csv("/home/messy92/Leo/Project_Gosin/SL+RL/Food_Generation/save_folder/" + save_result_name1)

reward_df = pd.read_csv("/home/messy92/Leo/Project_Gosin/SL+RL/Food_Generation/save_folder/" + save_result_name1)
plt.title(actor_mode + "-" + "LS-" + likelihood_sampling)
sns.lineplot(x = "Epoch", y = "Mean Reward", data = reward_df, hue = "condition")
plt.axhline(y = real_mean_nutrient_score, color = 'red', linestyle = ':')
plt.legend(bbox_to_anchor = (0, -0.2), loc = 2, borderaxespad = 0.)
plt.show()

# 정확도 플로팅
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
target_dat = mean_a_prob_score_vector_list_all
prob_df = pd.DataFrame(target_dat)
prob_df.columns = ["Epoch", "Prob", "Agent", "Network", "condition"]

prob_df['Epoch'] = pd.to_numeric(prob_df['Epoch'], errors = "coerce").astype(int)
prob_df['Prob'] = pd.to_numeric(prob_df['Prob'], errors = "coerce")

save_result_name2 = "LS:" + likelihood_sampling + "-CL:" + closed_loop + "-Network:" + actor_mode + "-measure2.csv"
prob_df.to_csv("/home/messy92/Leo/Project_Gosin/SL+RL/Food_Generation/save_folder/" + save_result_name2)

prof_df = pd.read_csv("/home/messy92/Leo/Project_Gosin/SL+RL/Food_Generation/save_folder/" + save_result_name2)
plt.title(actor_mode + "-" + "LS-" + likelihood_sampling)
sns.lineplot(x = "Epoch", y = "Prob", data = prob_df, hue = "condition")
plt.legend(bbox_to_anchor = (0, -0.2), loc = 2, borderaxespad = 0.)
plt.show()

# %%
# 생성 TEST
continue_ = 1
iter_num = 0
GenDiet_mean_reward_list = []
Gen_DF = pd.DataFrame([])

while(continue_):
    iter_num += 1

    if iter_num % 30 == 0:
        print('iter_num :', iter_num)
        print(tf.reduce_mean(GenDiet_mean_reward_list))

    hidden_state = Teacher2.initialize_hidden_state(BATCH_SIZE)
    init_tokens = np.array([1726] * list(tf_dataset)[0].shape[0]) 
    init_tokens = tf.convert_to_tensor(init_tokens, dtype = tf.int32)
    init_tokens = tf.reshape(init_tokens, shape = (-1, 1))

    inputs = copy.deepcopy(init_tokens) # SL이 학습하는 데이터
    gen_sequences = copy.deepcopy(inputs)

    '''
    [시작] 토큰 다음 토큰 생성
    '''
    outputs1, hidden_state = Teacher2(inputs, h_state = hidden_state)
    results = np.apply_along_axis(get_action, 1, outputs1, option = "prob") # 각 배치별로 output (= policy)에 근거한 행위와 그 행위확률 반환
    actions = results[:, 0] # 행위들만 반환
    actions = tf.reshape(actions, shape = (-1, 1))
    gen_sequences = np.hstack([gen_sequences, actions])
    inputs = copy.deepcopy(actions)

    a_hidden_state = Actor.initialize_hidden_state(BATCH_SIZE) # 각 모델의 hidden_state를 Teacher의 최초 hidden state로 선언
    for i in range(1, 16 - 2):

        if actor_mode == "RNN":
            outputs1, a_hidden_state = Actor(inputs, a_hidden_state)  # Actor
        elif actor_mode == "MLP":
            outputs1 = Actor(inputs)  # Actor

        results = np.apply_along_axis(get_action, 1, outputs1, option = "prob") # 각 배치별로 output (= policy)에 근거한 행위와 그 행위확률 반환
        actions = results[:, 0]         # 행위들만 반환

        actions = tf.reshape(actions, shape = (-1, 1))

        gen_sequences = np.hstack([gen_sequences, actions])
        inputs = copy.deepcopy(actions)

    gen_nutrient_state = np.apply_along_axis(get_score_vector, axis = 1, arr = gen_sequences, nutrient_data = nutrient_data)
    gen_per_sample_rewards = np.apply_along_axis(get_reward_ver2, axis = 1, arr = gen_nutrient_state, done = done)
    score_vector = tf.convert_to_tensor(gen_per_sample_rewards[:, 0], dtype = tf.float32)    # 순수 보상

    GenDiet_mean_reward_list = np.append(GenDiet_mean_reward_list, tf.reduce_mean(score_vector))

    gen_sentences = sequence_to_sentence(gen_sequences, food_dict) 
    gen_dataframe = pd.DataFrame(gen_sentences)
    gen_dataframe['Nutrition Score'] = score_vector

    Gen_DF = Gen_DF.append(gen_dataframe)

    if iter_num == 200:
        continue_ = 0

print(tf.reduce_mean(GenDiet_mean_reward_list))
sequence_to_sentence(gen_sequences, food_dict)

save_generated_diet = "LS:" + likelihood_sampling + "-CL:" + closed_loop + "-Network:" + actor_mode + "-genDiet.csv"
Gen_DF.to_csv("/home/messy92/Leo/Project_Gosin/SL+RL/Food_Generation/save_folder/" + save_generated_diet)

# %%
# 데이터 합치기
path = "/home/messy92/Leo/Project_Gosin/SL+RL/Food_Generation/save_folder"
measure = input("Merge by which measure: Mean Reward or Prob")
network = input("Which Network Do You Want? : MLP or RNN")
total_df, all_files = merge_result_files(path, measure, network)

# 후처리 : "Agent" 컬럼 값이 "t"인 값들 제거
postp_total_df = total_df[total_df["Agent"] != "T"]

# 그리기
plt.title("-".join(np.array(postp_total_df.iloc[0, 4:])))
# sns.lineplot(x = "Epoch", y = measure, data = postp_total_df, hue = "Agent", palette = color_dict)
if measure == "Mean Reward":
    real_mean_nutrient_score = return_nutrient_score(diet_data_np, nutrient_data)
    plt.axhline(y = real_mean_nutrient_score, color = 'red', linestyle = ':')
sns.lineplot(x = "Epoch", y = measure, data = postp_total_df, hue = "condition")
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.show()

# %%
