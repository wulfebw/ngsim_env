
import numpy as np
import os 
import sys
import tensorflow as tf

backend = 'Agg' if sys.platform == 'linux' else 'TkAgg'
import matplotlib
matplotlib.use(backend)
import matplotlib.pyplot as plt

import hgail.misc.simulation

import utils

def visualize_trajectories(output_dir, trajs, length):

    rmses = []
    for traj in trajs:
        if len(traj['rmse']) == length:
            rmses.append(traj['rmse'])
    rmses = np.array(rmses)
    mean = np.mean(rmses, axis=0)
    bound = np.std(rmses, axis=0) / np.sqrt(len(rmses)) / 2
    x = range(len(mean))
    plt.fill_between(x, mean - bound, mean + bound, alpha=.4, color='blue')
    plt.plot(x, mean, c='blue')
    plt.xlabel('timesteps')
    plt.ylabel('rmse')
    plt.title('mean rmse: {}'.format(np.mean(rmses)))
    plt.show()

def write_trajectories(filepath, trajs):
    np.savez(filepath, trajs=trajs)

def load_trajectories(filepath):
    return np.load(filepath)['trajs']

def collect_trajectories(
        args, 
        params,
        n_traj=100):

    # build components
    env, _, _ = utils.build_ngsim_env(args, alpha=0.)
    policy = utils.build_policy(args, env)

    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())

        # then load parameters
        policy.set_param_values(params['policy'])
        normalized_env = hgail.misc.utils.extract_normalizing_env(env)
        if normalized_env is not None:
            normalized_env._obs_mean = params['normalzing']['obs_mean']
            normalized_env._obs_var = params['normalzing']['obs_var']

        # collect trajectories
        trajs = hgail.misc.simulation.collect_trajectories(
            n_traj=n_traj,
            env=env,
            policy=policy,
            max_steps=1000 # overshoot and let it terminate
        )

        hgail.misc.simulation.simulate(env, policy, args.env_H, render=True)

    return trajs

if __name__ == '__main__':
    # load information relevant to the experiment
    exp_dir = '../../data/experiments/NGSIM-gail/'
    args_filepath = os.path.join(exp_dir, 'imitate/log/args.npz')
    args = np.load(args_filepath)['args'].item()
    params_filepath = os.path.join(exp_dir, 'imitate/log/itr_109.npz')
    params = hgail.misc.utils.load_params(params_filepath)

    # replace ngsim_filename with different file for cross validation
    # args.ngsim_filename = 'trajdata_i101_trajectories-0805am-0820am.txt'
    args.ngsim_filename = 'trajdata_i101_trajectories-0820am-0835am.txt'

    # validation setup 
    validation_dir = os.path.join(exp_dir, 'imitate', 'validation')
    utils.maybe_mkdir(validation_dir)
    output_filepath = os.path.join(validation_dir, '{}_trajectories.npz'.format(args.ngsim_filename.split('.')[0]))

    trajs = collect_trajectories(args, params, n_traj=1)
    write_trajectories(output_filepath, trajs)

    trajs = load_trajectories(output_filepath)
    visualize_trajectories(validation_dir, trajs, length=args.env_H)
