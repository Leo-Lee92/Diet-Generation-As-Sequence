# Diet-Generation-As-Sequence
This repository covers the studies including:
1. *Diet Planning with Machine Learning: Teacher-forced REINFORCE for Composition Compliance with Nutrition Enhancement* (KDD 21' accepted)
2. *(temporary title) Avoid learning poor-quality existing records: An OR-Experts-ML (ORxML) collaboration framework for synthesizing high quality non-existing data of human task* (on-going)

The first study deals with our novel algortihm which combines REINFORCE algorithm with teacher-forcing technique to overcome the limitation, called exposure-bias (aka collapse mode) while using on-policy algorithm.

You can use our algorithm by directly running linux code:

```bash
python3 main.py --language=korean  --add_breakfast=True --network=GRU --attention=False --embed_dim=128 --fc_dim=64 --learning=off-policy --policy=target --beta=True --buffer=True --buffer_size=30 --buffer_update=10 --num_epochs=20000 --lr=1e-3
```
where each arguments are parsed with command ```--args1 = option```. You can customize the option and can do the ablation study. The options of arguments are as below. Read the description written in ```kwargs``` dictionary, and change options according to the descriptions. 

```python
# Variable Parameter Initialization Ver.2 (If you pass the paremeters at Jupyter interactive level.)
language = 'korean'
add_breakfast = True
network = 'GRU' 
attention =  False 
embed_dim = 128
fc_dim = 64
learning = 'off-policy'
policy = 'random'
beta = True
buffer = True
buffer_size = 5
buffer_update = 5
num_epochs = 10000
lr = 1e-3
kwargs = {
    # This parameter can be 'korean' or 'english'.
    # We recommend you to use 'korean' only as we found the execution unstable when to use 'english'.
    'language' : language,

    # This parameter can be True or False.
    # If it is set to be True, then the length of sequence is 21.
    # On the other hand, if it is set to be False, the length of sequence is 16.
    # The mode of get_reward_ver2() changes according to this parameter as well.
    ## Possible parameter list = [True, False]
    'add_breakfast' : add_breakfast,

    # This parameter can be 'GRU' or 'LSTM'.
    ## Possible parameter list = ['GRU', 'LSTM']
    'fully-connected_layer' : network, 
    
    # This parameter can be True or False. 
    # If 'attention' = False, attention is not applied in the model.
    ## Possible parameter list = [True, False]
    'attention': attention, 

    # This parameter represents the size of embedding layer (i.e., the number of neurons in the embedding layer).
    ## Therefore, any integer value can be used for this parameter.
    'embed_dim': embed_dim,

    # This parameter represents the size of fully-connected layer (i.e., the number of neurons in the fully-connected layer).
    ## Therefore, any integer value can be used for this parameter.
    'fc_dim': fc_dim,

    # This parameter can be 'on-policy' or 'off-policy'. 
    # If 'policy_tpye' = 'on-policy', then target-policy and behavior policy is same (i.e., REINFORCE algorithm is run), which means actions are sampled according to target-policy distribution.
    # If 'policy_type' = 'off-policy', then behavior policy is given as the stream of real data, which means actions are sampled according to data distribution.
    ## Possible parameter list = ['on-policy', 'off-policy']
    'learning': learning,

    # This parameter can be random-policy, greedy-policy, and target-policy.
    ## Possible paramter list = [random, greedy, target]
    'policy': policy,

    # This parameter can be True or False.
    ## Possible parameter list = [True, False]
    'use_beta': beta,

    # This parameter can be True or False. 
    # If 'use_buffer' = True, then the target data is replaced by synthetic data in stochastic way.
    # On the other hand, if 'use_buffer' == False, then the target data is fixed in constant.
    ## Possible parameter list = [True, False]
    'use_buffer': buffer,

    # Else parameters.
    'buffer_size' : buffer_size,
    'buffer_update' : buffer_update,
    'num_epochs' : num_epochs,
    'lr' : lr, 
    'num_tokens' : len(food_dict),
    'batch_size' : BATCH_SIZE
}
```


---
P.S) The code structure given in this repository is as below:

![image](https://user-images.githubusercontent.com/61273017/121123076-8db69900-c85d-11eb-9b45-11424b3e14b8.png)