# Diet-Generation-As-Sequence
This repository covers the studies including:
1. *Diet Planning with Machine Learning: Teacher-forced REINFORCE for Composition Compliance with Nutrition Enhancement* (KDD 21' accepted)
2. *(temporary title) Avoid learning poor-quality existing records: An OR-Experts-ML (ORxML) collaboration framework for synthesizing high quality non-existing data of human task* (on-going)

The first study deals with our novel algortihm which combines REINFORCE algorithm with teacher-forcing technique to overcome the limitation, called exposure-bias (aka collapse mode) while using on-policy algorithm.

You can use our algorithm by directly running linux code:

```
python3 main.py --language=korean  --add_breakfast=True --network=GRU --attention=False --embed_dim=128 --fc_dim=64 --learning=off-policy --policy=target --beta=True --buffer=True --buffer_size=30 --buffer_update=10 --num_epochs=20000 --lr=1e-3
```
where each arguments are parsed with command ```--args1 = option```. You can customize the option and can do the ablation study. Please refer to ```Preprocessing.py``` file to check the list of possible options. We comminted out the list in this file. 

---
P.S) The code structure given in this repository is as below:

![image](https://user-images.githubusercontent.com/61273017/121123076-8db69900-c85d-11eb-9b45-11424b3e14b8.png)