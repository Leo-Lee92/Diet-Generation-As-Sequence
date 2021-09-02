# Diet-Generation-As-Sequence
***<span style="color:crimson"> ! Note that you have to make clear your anaconda environment shoulde be properly preprared and set right as provided in ```settings.yml``` file. Otherwise, our code may not work.</span>***

This repository covers the studies including:
1. *Diet Planning with Machine Learning: Teacher-forced REINFORCE for Composition Compliance with Nutrition Enhancement* (KDD 21' accepted)
2. *K--MIND dataset for diet planning and healthcare research: Dataset creation using combinatorial optimization and controllable generation with domain experts* (under review in Neurips 21')

The first study deals with our novel algortihm which combines REINFORCE algorithm with teacher-forcing technique to overcome the typical limitation of neural machine translation called exposure-bias which is caused by the nature of online learning (on-policy learning).

You can use our algorithm by directly running linux code:

```bash
python3 main.py --network=GRU --attention=True --embed_dim=128 --fc_dim=64 --learning=off-policy --policy=target --beta=True --buffer=True --buffer_size=30 --buffer_update=10 --num_epochs=40000 --lr=1e-3
```
where each arguments are parsed with command ```--args = option```. You can customize the option and can do the ablation study. The options of arguments are as below. Read the description written in ```kwargs``` dictionary, and change options according to the descriptions. 

```python
# Variable Parameter Initialization Ver.2 (If you pass the paremeters at Jupyter interactive level.)
network = 'GRU' 
attention =  True 
embed_dim = 128
fc_dim = 64
learning = 'off-policy'
policy = 'target'
beta = True
buffer = True
buffer_size = 30
buffer_update = 10
num_epochs = 40000
lr = 1e-3
kwargs = {
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
# Code schema

The overall code schema is given as below:
<!-- ![image](https://user-images.githubusercontent.com/61273017/121123076-8db69900-c85d-11eb-9b45-11424b3e14b8.png) -->
<img src = "https://user-images.githubusercontent.com/61273017/121123076-8db69900-c85d-11eb-9b45-11424b3e14b8.png">


# Examples of **Utilities** 
## (1) language settings
- For users who are not familiar with Korean food and language, we provide a function that changes name of all menus from Korean to English. This function requires to have a file that matches every menus in korean to its correspondings in english. We provide this file in *`./Data (new)/translated_menu_name (수정).csv`*. Below code provides a function to translate Korean to English.
```python
# Translate a diet with korean into a diet with english.
translate_book = pd.read_csv('./Data (new)/translated_menu_name (수정).csv', encoding='CP949')
translate_dict = dict(list(zip(translate_book['kor'], translate_book['eng'])))
def kor_to_eng(sentence_list, refer = translate_dict):

    gen_food_list = []

    for i, val in enumerate(sentence_list):
        each_food = []

        for j, val2 in enumerate(val):
            each_food.append(translate_dict[val2])

        gen_food_list.append(each_food)

    return(gen_food_list)
```
## (2) reverse tokenization
- During training and evaluation, diet sequence is an array of tokens where each token encodes each menu. However, a tokenization is not understandable therefore we need to decode them back to its original state, a menu. Below code provides a function to decode token to menu.
```python
# Transform a sequence (diet encoded with tokens) to sentence (diet with menus).
def sequence_to_sentence(sequence_list, food_dict):
    gen_food_list = []

    for i, val in enumerate(sequence_list):
        each_food = []

        for j, val2 in enumerate(val):
            each_food.append(food_dict[val2])

        gen_food_list.append(each_food)

    return(gen_food_list)
```
# **Evaluations**
## (1) Save results
`evaluation.py` provides and analyzes the results in various perspectives. First of all, as the generated diets by ML module is automatically saved in local directory *`./results/gen_diets(eng).csv`*. Refer to below code.
```python
# Save the generated result
## korean version
pd.DataFrame(kor_to_eng(sequence_to_sentence(gen_seqs, food_dict), translate_dict)).to_csv('./results/gen_diets(eng).csv', encoding='CP949')
## english version
pd.DataFrame(sequence_to_sentence(gen_seqs, food_dict)).to_csv('./results/gen_diets(kor).csv', encoding='CP949')
```

## (2) Compute RDI scores
And, RDI score of both expert- ang ML-generated diets can be calculated as below. Note that the functions `get_score_pandas()` and `get_reward_ver2()` are imported from `util.py`.
```python
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
```
Make sure that *`food_dict`* and *`nutrient_data`* are dictionary of foods (dictionary of menus) and menu-nutrient data respectively, while *`x`* and *`gen_seqs`* are the expert- and ML-generated diet sequences.

## (3) Visualize Nutrient distributions
In addition, you can visually check what types of nutrient and how much samples achieve the nutrition requirements using below code
```python
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
```
and the visualized result is:

<img src = "https://user-images.githubusercontent.com/61273017/131641545-471f0562-4cc9-485a-b5d3-cf6d00d4a456.png" width = 650>


# **Usage of dietkit** to check *nutrition*, *ingredient* of generated diets. #
Meanwhile, lets attempt to analyze the level of nutrition and types of ingredients in detail at the level of each diet. 

First, select the top 200 pairs with the farthest jaccard distance between the diets, when each pair consists of a expert- and ML-generated diet. Here, when the expert-generated and ML-generated diets have far jaccard distance, it implies that the diets from expert and ML consist of different composition, meaning the usage of menus, ingredients, etc. is far different. 

To do this, we provide function `get_most_change_sequence()` that works to filter *`top_n`* pairs of the farthest jaccrad distance. Specifically, this function returns indices of the pairs.
```python
# Get top 200 expert-ML generated diet pair.
top_change_sequence = get_most_change_sequence(x, gen_seqs, top_n = 200)
top_chnage_idx = top_change_sequence['example_idx']
```

Second, let us split and store each pair of given indices (*`example_idx`*) into *`sample_diets_expert`* and *`sample_diets_ML`* variables. Each of variables stands for the diet sequences generated by expert and ML, respectively. 
```python
# Get samples of expert- and ML-generated diet.
## get sample without BOS and EOS tokens.
sample_diets_expert = []
sample_diets_ML = []
for i in top_chnage_idx:
    sample_diets_expert.append(kor_to_eng(sequence_to_sentence([x.numpy()[i, 1:x.shape[1] - 1]], food_dict))[0])
    sample_diets_ML.append(kor_to_eng(sequence_to_sentence([gen_seqs[i, 1:gen_seqs.shape[1] - 1]], food_dict))[0])
```

Third, install *`dietkit`* package. We have to install this package because it provides various functions for diet planning. The installation process is as below:
## (1) Download **dietkit**
```bash
pip install dietkit
```

After installing *`dietkit`*, import and get a dictionary of menus *`menus_dict`*, using embedded function `load_menu()`.
```python
# Import dietkit
from dietkit.loader import *

# Load menus
menus_dict = load_menu()
```

## (2) Analysis of diets
To use embedded functions provided by *`dietkit`*, we have to set the diet data as an object of dietkit. A function `objectification()`:
```python
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

    ## Change diet as an object_diet, which is able to apply the embedded functions of dietkit. Otherwise you cannot use dietkit.
    object_diet_sequences = []
    for diet_idx, diet in enumerate(sample_diet_sequences):
        object_diet_sequences.append([ menus_dict[menu] for menu in diet ])

    object_diet_sequences = Diet(dict(list(zip(range(len(object_diet_sequences)), object_diet_sequences[:]))))
    
    return object_diet_sequences
```
makes it available according to:
```python
object_diet_expert = objectivication(sample_diets_expert, menus_dict)
object_diet_ML = objectivication(sample_diets_ML, menus_dict)
```

As shown above, *`object_diet_expert`* and *`object_diet_ML`* are the dietkit object of *`sample_diets_expert`* and *`sample_diets_ML`*, respectively. Now, we can obtain various information of the diets with a levarge of **dietkit**, for example, 
```python
# expert-generated diets
object_diet_expert.plan             # return dictionary of diets where each diet is a list of menus.
object_diet_expert.menu_category()  # return category-level of each menu of diet sequences. 
object_diet_expert.nutrition        # return nutrition of given diet
object_diet_expert.ingredient       # return list of diets where each diet is a list of ingredients.

# ML-generated diets
object_diet_ML.plan             # return dictionary of diets where each diet is a list of menus.
object_diet_ML.menu_category()  # return category-level of each menu of diet sequences. 
object_diet_ML.nutrition        # return nutrition of given diet
object_diet_ML.ingredient       # return list of diets where each diet is a list of ingredients.
```

## (3) Visualization with TSNE
Now, we have to tokenize the ingredients as integer embeddings. To do that, you need to use `ingredient_tokenization` function.
```python
import copy
def ingredient_tokenization(diet_ingred_sequences, ingred_dict):
    diet_ingred_tokenize = []
    for diet_idx, diet in enumerate(diet_ingred_sequences):
        diet_ingred_tokenize.append([ingred_dict[ingred] for ingred in diet])

    return diet_ingred_tokenize
```

```python
# Get a list of diet sequences, i.e., diet_ingred_sequences, where each diet consists of ingredients rather than menus.
diet_ingred_sequences_expert = list(object_diet_expert.ingredient.values())
diet_ingred_sequences_ML = list(object_diet_ML.ingredient.values())

# Combine diet_ingred_sequences of expert and ML, and get a total number of ingredients used.
diet_ingred_sequences_total = list(np.concatenate(diet_ingred_sequences_expert + diet_ingred_sequences_ML).flat)
diet_ingred_sequences_total = set(diet_ingred_sequences_total)
num_of_ingreds = len(diet_ingred_sequences_total)

# Get the ingredient dictionary
ingred_dict = [ [ingred, ingred_idx] for ingred_idx, ingred in enumerate(diet_ingred_sequences_total)]
ingred_dict = dict(ingred_dict)

diet_tokenized_ingred_ML = ingredient_tokenization(diet_ingred_sequences_ML, ingred_dict)
diet_tokenized_ingred_expert = ingredient_tokenization(diet_ingred_sequences_expert, ingred_dict)
```

```python
# Get onehot matrix of diet_ingred_sequences with respect to both expert- and ML- generated.
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words = num_of_ingreds)
tokenizer.fit_on_sequences(diet_tokenized_ingred_ML)
diet_onehot_ingred_expert = tokenizer.sequences_to_matrix(diet_tokenized_ingred_expert)
diet_onehot_ingred_ML = tokenizer.sequences_to_matrix(diet_tokenized_ingred_ML)
```

```python
# Get TSNE embedding space of both expert- and ML-generated diet_ingred_sequences  
from sklearn.manifold import TSNE
diet_onehot_ingred_total = np.concatenate([diet_onehot_ingred_expert, diet_onehot_ingred_ML], axis = 0)
diet_ingred_embedding = TSNE(n_components = 2).fit_transform(diet_onehot_ingred_total)

xs = diet_ingred_embedding[:,0] # x coordinate
ys = diet_ingred_embedding[:,1] # y coordinate
labels = np.concatenate([ np.repeat(i, diet_onehot_ingred_expert.shape[0]) for i in range(2) ]).ravel() # label

# Visualize
import seaborn as sns
tsne_values = diet_ingred_embedding
tsne_df = pd.DataFrame(tsne_values)
tsne_df['method'] = labels
tsne_df.columns = ["x", "y", "method"]

colors = ["#d62728", "#1f77b4"]
sns.set_palette(sns.color_palette(colors))
sns.scatterplot(data = tsne_df, x = "x", y = "y", hue = "method", style = "method", s = 50, alpha = .8)
```

# Analysis of **Attention** Maps
```python
'''
Analysis with Attention map
'''
# Attention Image 뽑는 코드 (evaluation.py로 옮겨주기)
# Define a single batch sample.
x = list(tf_dataset_for_eval)[batch]             

# Generate sequences (i.e., synthetic diets).
gen_seqs, atts = diet_generator.inference(x, return_attention = True)     

# Get attention maps of examples where the translated diet has highest distance with source diet. The distance is defined as jaccard distance.
top_change_sequence = get_most_change_sequence(x, gen_seqs, top_n = 5)
top_chnage_idx = top_change_sequence['example_idx']
for i in top_chnage_idx:
    plot_attention(i, atts[i], x.numpy()[i], gen_seqs[i], food_dict, translate_dict, language = 'eng')
```

## **(1) english version**

<img src = "https://user-images.githubusercontent.com/61273017/131300637-a335292c-40e3-452c-944d-134f1d41f376.png" width = "650"><img src = "https://user-images.githubusercontent.com/61273017/131300641-c3e451da-7ab0-4993-a3d2-2cfb42fab18a.png" width = "650"><img src = "https://user-images.githubusercontent.com/61273017/131300643-291616c2-6528-4563-b69b-720e46376f83.png" width = "650"><img src = "https://user-images.githubusercontent.com/61273017/131300646-0c330451-a2e1-4a4d-9759-1b21684d41a6.png" width = "650">

## **(2) korean version**

<img src = https://user-images.githubusercontent.com/61273017/131300654-f1f53333-4c9c-4d2e-a6a6-c00ec5d66f55.png width = "650"><img src = https://user-images.githubusercontent.com/61273017/131300655-2361daf7-c297-4b28-ad2d-0d88d4d390a3.png width = "650"><img src = https://user-images.githubusercontent.com/61273017/131300656-48f757b1-b552-44c4-9e32-460c0f6872a0.png width = "650"><img src = https://user-images.githubusercontent.com/61273017/131300650-62093693-a5b0-4d9e-91d4-35d2ca8aeb3e.png width = "650"> 

