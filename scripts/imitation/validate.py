
import multiprocessing as mp
import numpy as np
import os 
import sys
import tensorflow as tf

backend = 'Agg' if sys.platform == 'linux' else 'TkAgg'
import matplotlib
matplotlib.use(backend)
import matplotlib.pyplot as plt

from context_timer import ContextTimer

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
        n_traj, 
        trajlist,
        pid=1,
        env_fn=utils.build_ngsim_env,
        policy_fn=utils.build_policy,
        max_steps=200):
    env, _, _ = env_fn(args, alpha=0.)
    policy = policy_fn(args, env)
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
        for traj_idx in range(n_traj):
            sys.stdout.write('\rpid: {} traj: {} / {}'.format(pid, traj_idx + 1, n_traj))
            traj = hgail.misc.simulation.simulate(env, policy, max_steps)
            trajlist.append(traj)

    return trajlist

def collect_hgail_trajectories(
        args,  
        params, 
        n_traj, 
        trajlist,
        pid=1,
        env_fn=utils.build_ngsim_env,
        hierarchy_fn=utils.build_hierarchy,
        max_steps=200):
    env, _, _ = env_fn(args, alpha=0.)
    hierarchy = hierarchy_fn(args, env)
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())

        # then load parameters
        for i, level in enumerate(hierarchy):
            level.algo.policy.set_param_values(params[i]['policy'])
        policy = hierarchy[0].algo.policy
        normalized_env = hgail.misc.utils.extract_normalizing_env(env)
        if normalized_env is not None:
            normalized_env._obs_mean = params['normalzing']['obs_mean']
            normalized_env._obs_var = params['normalzing']['obs_var']

        # collect trajectories
        for traj_idx in range(n_traj):
            sys.stdout.write('\rpid: {} traj: {} / {}'.format(pid, traj_idx + 1, n_traj))
            traj = hgail.misc.simulation.simulate(env, policy, max_steps)
            trajlist.append(traj)

    return trajlist

def parallel_collect_trajectories(
        args,
        params,
        n_traj,
        n_proc,
        collect_fn=collect_trajectories):

    manager = mp.Manager()
    trajlist = manager.list()
    
    # just force it to be divisible 
    assert n_traj % n_proc == 0
    n_traj_proc = n_traj // n_proc

    # pool 
    pool = mp.Pool(processes=n_proc)

    # start collection
    results = []
    for pid in range(n_proc):
        res = pool.apply_async(
            collect_fn,
            args=(args, params, n_traj_proc, trajlist, pid)
        )
        results.append(res)

    # wait for the processes to finish
    [res.get() for res in results]

    return trajlist

def collect_gail(filename=''):
    # load information relevant to the experiment
    exp_dir = '../../data/experiments/NGSIM-infogail/'
    args_filepath = os.path.join(exp_dir, 'imitate/log/args.npz')
    args = np.load(args_filepath)['args'].item()
    params_filepath = os.path.join(exp_dir, 'imitate/log/itr_1999.npz')
    params = hgail.misc.utils.load_params(params_filepath)
    n_traj = 1000
    n_proc = 10

    # replace ngsim_filename with different file for cross validation
    if filename != '':
        args.ngsim_filename = filename

    # validation setup 
    validation_dir = os.path.join(exp_dir, 'imitate', 'validation')
    utils.maybe_mkdir(validation_dir)
    output_filepath = os.path.join(validation_dir, '{}_trajectories.npz'.format(args.ngsim_filename.split('.')[0]))

    with ContextTimer():
        trajs = parallel_collect_trajectories(args, params, n_traj, n_proc)

    write_trajectories(output_filepath, trajs)

def collect_hgail(filename=''):
    exp_dir = '../../data/experiments/NGSIM-hgail/'
    args_filepath = os.path.join(exp_dir, 'imitate/log/args.npz')
    args = np.load(args_filepath)['args'].item()
    params_filepath = os.path.join(exp_dir, 'imitate/log/itr_6.npz')
    params = hgail.misc.utils.load_params(params_filepath)
    n_traj = 1000
    n_proc = 10

    # replace ngsim_filename with different file for cross validation
    if filename != '':
        args.ngsim_filename = filename

    # validation setup 
    validation_dir = os.path.join(exp_dir, 'imitate', 'validation')
    utils.maybe_mkdir(validation_dir)
    output_filepath = os.path.join(validation_dir, '{}_trajectories.npz'.format(args.ngsim_filename.split('.')[0]))

    with ContextTimer():
        trajs = parallel_collect_trajectories(args, params, n_traj, n_proc, collect_fn=collect_hgail_trajectories)

    write_trajectories(output_filepath, trajs)

if __name__ == '__main__':
    filenames = [
        "trajdata_i80_trajectories-0400-0415.txt",
        "trajdata_i80_trajectories-0500-0515.txt",
        "trajdata_i80_trajectories-0515-0530.txt",
        "trajdata_i101_trajectories-0805am-0820am.txt",
        "trajdata_i101_trajectories-0820am-0835am.txt",
        "trajdata_i101_trajectories-0750am-0805am.txt"
    ]
    for fn in filenames:
        # collect_gail(fn)
        collect_hgail(fn)
