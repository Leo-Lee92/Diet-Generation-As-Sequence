# %%
# import Preprocessing
# import Model
from Preprocessing import *
from Model import *
import tensorflow as tf
import numpy as np
import copy

if __name__ == "__main__":
    encoder = Encoder(len(food_dict))
    for x in tf_dataset:
        enc_output, enc_state = encoder(x)
        print(enc_output)


# %%
# encoder = Encoder(len(food_dict))
# tmp = encoder(tf.expand_dims(diet_data_np[0, :], axis = 0))