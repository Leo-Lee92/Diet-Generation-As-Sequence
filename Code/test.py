# %%

# def main():
num_epochs = 5000
encoder = Encoder(len(food_dict), BATCH_SIZE)
decoder = Decoder(len(food_dict))

for epoch in range(num_epochs):
    for x in tf_dataset:
        enc_hidden = encoder.initialize_hidden_state()
        batch_loss = train_step(x, enc_hidden)
        print('epoch : {}, batch_loss : {}'.format(epoch, batch_loss)) 
# %%
# generation 파트
true_total_reward = 0
gen_total_reward = 0
for i in range(len(list(tf_dataset)[0])):
    sample_input = list(tf_dataset)[0][i]
    sample_input = tf.reshape(sample_input, shape = (1, -1))
    sample_enc_hidden = encoder.initialize_hidden_state()
    _, sample_enc_hidden = encoder(sample_input, sample_enc_hidden)

    sample_targets_x = np.array([1726])
    sample_dec_hidden = copy.deepcopy(sample_enc_hidden)

    sample_seqs = np.array([])
    sample_seqs = np.append(sample_seqs, sample_targets_x)

    for j in range(15):
        sample_outputs, sample_dec_hidden = decoder(sample_targets_x, sample_dec_hidden)
        # next_token = tf.argmax(sample_outputs)    # greedy selection
        next_token = np.random.choice(len(sample_outputs), p = np.array(sample_outputs))    # stochastic selection
        sample_targets_x = copy.deepcopy(tf.expand_dims(next_token, 0))
        sample_dec_hidden = tf.reshape(sample_dec_hidden, shape = (1, -1))

        sample_seqs = np.append(sample_seqs, next_token)

    print(' ')
    print(' 정답 :', sequence_to_sentence(np.array(sample_input), food_dict))
    print(' 생성 :', sequence_to_sentence([sample_seqs], food_dict))
    gen_reward = get_reward_ver2(get_score_vector(sample_seqs, nutrient_data), 0)[0]
    true_reward = get_reward_ver2(get_score_vector(sample_input[0], nutrient_data), 0)[0]
    print(' ')
    print(' 정답의 보상 :', true_reward)
    print(' 생성의 보상 :', gen_reward)

    true_total_reward += true_reward
    gen_total_reward += gen_reward

true_mean_reward = true_total_reward / len(list(tf_dataset)[0])
gen_mean_reward = gen_total_reward / len(list(tf_dataset)[0])

print('true_mean_reward :', true_mean_reward)
print('gen_mean_reward :', gen_mean_reward)


# %%
# 영양소 벡터로 바꿔주기
nutrient_array = np.empty((0, nutrient_data.shape[1] - 1))

for x in tf_dataset:
    tmp = np.apply_along_axis(get_score_vector, arr = x, axis = 1, nutrient_data = nutrient_data)
    nutrient_array = np.vstack([nutrient_array, tmp])    