'''
default hyperparameters for training
these are build as args to allow for command line options 
these args are also saved along with parameters during training to 
allow for rebuilding everything with the same settings
'''

import argparse
import numpy as np

def str2bool(v):
    if v.lower() == 'true':
        return True
    return False

def parse_args():
    parser = argparse.ArgumentParser()
    # logistics
    parser.add_argument('--exp_name', type=str, default='NGSIM-gail')
    parser.add_argument('--params_filepath', type=str, default='')
    parser.add_argument('--expert_filepath', type=str, default='../../data/trajectories/ngsim.h5')
    parser.add_argument('--vectorize', type=str2bool, default=False)
    parser.add_argument('--n_envs', type=int, default=50)
    parser.add_argument('--normalize_clip_std_multiple', type=float, default=10.)

    # env
    parser.add_argument('--ngsim_filename', type=str, default='trajdata_i101_trajectories-0750am-0805am.txt')
    parser.add_argument('--env_H', type=int, default=200)
    parser.add_argument('--env_primesteps', type=int, default=50)

    # reward handler
    parser.add_argument('--reward_handler_max_epochs', type=int, default=100)
    parser.add_argument('--reward_handler_recognition_final_scale', type=float, default=.2)

    # policy 
    parser.add_argument('--use_infogail', type=str2bool, default=True)
    parser.add_argument('--policy_mean_hidden_layer_dims', nargs='+', default=(128,128,64))
    parser.add_argument('--policy_std_hidden_layer_dims', nargs='+', default=(128,64))

    # critic
    parser.add_argument('--use_critic_replay_memory', type=str2bool, default=True)
    parser.add_argument('--n_critic_train_epochs', type=int, default=55)
    parser.add_argument('--critic_learning_rate', type=float, default=.0002)
    parser.add_argument('--critic_dropout_keep_prob', type=float, default=.8)
    parser.add_argument('--gradient_penalty', type=float, default=2.)
    parser.add_argument('--critic_grad_rescale', type=float, default=40.)
    parser.add_argument('--critic_batch_size', type=int, default=1000)
    parser.add_argument('--critic_hidden_layer_dims', nargs='+', default=(128,128,64))

    # recognition
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--n_recognition_train_epochs', type=int, default=30)
    parser.add_argument('--scheduler_k', type=int, default=20)
    parser.add_argument('--recognition_learning_rate', type=float, default=.0005)
    parser.add_argument('--recognition_hidden_layer_dims', nargs='+', default=(128,64))

    # gail
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--trpo_step_size', type=float, default=.025)
    parser.add_argument('--n_itr', type=int, default=2000)
    parser.add_argument('--max_path_length', type=int, default=1000)
    parser.add_argument('--discount', type=float, default=.95)

    # parse and return
    args = parser.parse_args()
    return args
