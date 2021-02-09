# %%
# Vanila
## 가장 최신 체크포인트 불러오기
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
checkpoint.restore(tf.train.latest_checkpoint('pretraining_1'))

true_total_reward = 0
gen_total_reward = 0
for i in range(len(list(tf_dataset)[0][0])):
# for i in range(1, 3):
    sample_input = list(tf_dataset)[0][0][i]
    sample_input = tf.reshape(sample_input, shape = (1, -1))
    
    sample_enc_hidden = encoder.initialize_hidden_state()
    sample_enc_output, sample_enc_hidden = encoder(sample_input, sample_enc_hidden)

    sample_targets_x = np.array([1726])

    # 두 인코더의 컨텍스트 벡터 연결해주기
    sample_dec_hidden = copy.deepcopy(sample_enc_hidden)

    sample_targets_x = np.array([1726])

    sample_seqs = np.array([])
    sample_seqs = np.append(sample_seqs, sample_targets_x)

    # 어텐션 플롯팅을 위한 어텐션 행렬 정의
    attention_plot = np.zeros((sample_input.shape[1] - 1, sample_input.shape[1]))
    for j in range(15):
        sample_outputs, sample_dec_hidden, attention_weights = decoder(sample_targets_x, sample_dec_hidden, sample_enc_output)
        next_token = tf.argmax(sample_outputs)    # greedy selection
        # next_token = np.random.choice(len(sample_outputs), p = np.array(sample_outputs))    # stochastic selection
        sample_targets_x = copy.deepcopy(tf.expand_dims(next_token, 0))
        sample_dec_hidden = tf.reshape(sample_dec_hidden, shape = (1, -1))

        sample_seqs = np.append(sample_seqs, next_token)

        attention_weights = tf.reshape(attention_weights, shape = (-1, ))
        attention_plot[j] = attention_weights.numpy()

    real_sentence = sequence_to_sentence(np.array(sample_input), food_dict)[0]
    pred_sentence = sequence_to_sentence([sample_seqs], food_dict)[0]

    print(' ')
    print(' 정답 :', real_sentence)
    print(' 생성 :', pred_sentence)
    plot_attention(attention_plot, real_sentence, pred_sentence)

    gen_reward = get_reward_ver2(get_score_vector(sample_seqs, nutrient_data), 0)[0]
    true_reward = get_reward_ver2(get_score_vector(sample_input[0], nutrient_data), 0)[0]
    print(' ')
    print(' 정답의 보상 :', true_reward)
    print(' 생성의 보상 :', gen_reward)

    '''
    어텐션 결과 확인
    '''
    sample_input

    true_total_reward += true_reward
    gen_total_reward += gen_reward

true_mean_reward = true_total_reward / len(list(tf_dataset)[0])
gen_mean_reward = gen_total_reward / len(list(tf_dataset)[0])

print('true_mean_reward :', true_mean_reward)
print('gen_mean_reward :', gen_mean_reward)
