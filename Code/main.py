# %%
def initialize_setting():

    # GPU setting
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 2048)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # System Path setting
    import sys
    sys.path.append("/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/Proposed")
    import os    

from util import *
def get_params():

    import argparse
    parser = argparse.ArgumentParser(description='receive the parameters')
    parser.add_argument('--language', type=str, required=True, help='type: korean or english')
    parser.add_argument('--add_breakfast', type=str2bool, required=True, help='type: True or False')

    parser.add_argument('--network', type=str, required=True, help='type: GRU or LSTM')
    parser.add_argument('--attention', type=str2bool, required=True, help='type: True or False')
    parser.add_argument('--embed_dim', type=int, required=True, help='type: dim_size of embedding layer')
    parser.add_argument('--fc_dim', type=int, required=True, help='type: dim_size of fully-connected layer')

    parser.add_argument('--learning', type=str, required=True, help='type: off-policy or on-policy')
    parser.add_argument('--policy', type=str, required=True, help='type: target, greedy or random')

    parser.add_argument('--beta', type=str2bool, required=True, help='type: True or False')
    parser.add_argument('--buffer', type=str2bool, required=True, help='type: True or False')
    parser.add_argument('--buffer_size', type=int, required=True, help='type: buffer_size')
    parser.add_argument('--buffer_update', type=int, required=True, help='type: buffer_update_period')

    parser.add_argument('--num_epochs', type=int, required=True, help='type: num_epochs')
    parser.add_argument('--lr', type=float, required=True, help='type: learning_rate')

    global args
    args = parser.parse_args()

    return args

def main():
    initialize_setting()
    global_params = get_params()
    return global_params


# 만약 main.py를 run 한다면
if __name__ == "__main__":
    global_params = main()
    print('global_params :', global_params)
    from train import *





