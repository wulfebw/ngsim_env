
import argparse
import h5py
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
import hgail.misc.utils

import utils

parser = argparse.ArgumentParser(description="validation settings")
parser.add_argument('--nproc', type=int, default=1)

'''
This is about as hacky as it gets, but I want to avoid editing the rllab 
source code as much as possible, so it will have to do for now.

Add a reset(self, kwargs**) function to the normalizing environment
https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
'''
def normalize_env_reset_with_kwargs(self, **kwargs):
    ret = self._wrapped_env.reset(**kwargs)
    if self._normalize_obs:
        return self._apply_normalize_obs(ret)
    else:
        return ret

def add_kwargs_to_reset(env):
    normalize_env = hgail.misc.utils.extract_normalizing_env(env)
    if normalize_env is not None:
        normalize_env.reset = normalize_env_reset_with_kwargs.__get__(normalize_env)

'''end of hack, back to our regularly scheduled programming'''

def simulate(env, policy, max_steps, render=False, env_kwargs=dict()):
    traj = hgail.misc.simulation.Trajectory()
    x = env.reset(**env_kwargs)
    policy.reset()
    for step in range(max_steps):
        if render: env.render()
        a, a_info = policy.get_action(x)
        nx, r, done, e_info = env.step(a)
        traj.add(
            policy.observation_space.flatten(x), 
            a, 
            r, 
            a_info,
            e_info
        )
        if done: break
        x = nx
    return traj.flatten()

def write_trajectories(filepath, trajs):
    np.savez(filepath, trajs=trajs)

def load_trajectories(filepath):
    return np.load(filepath)['trajs']

def collect_trajectories(
        args,  
        params, 
        egoids, 
        trajdict,
        pid,
        env_fn,
        policy_fn,
        max_steps,
        use_hgail):
    env, _, _ = env_fn(args, alpha=0.)
    add_kwargs_to_reset(env)
    policy = policy_fn(args, env)
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())

        # then load parameters
        if use_hgail:
            for i, level in enumerate(policy):
                level.algo.policy.set_param_values(params[i]['policy'])
            policy = policy[0].algo.policy
        else:
            policy.set_param_values(params['policy'])
        normalized_env = hgail.misc.utils.extract_normalizing_env(env)
        if normalized_env is not None:
            normalized_env._obs_mean = params['normalzing']['obs_mean']
            normalized_env._obs_var = params['normalzing']['obs_var']

        # collect trajectories
        nids = len(egoids)
        for i, egoid in enumerate(egoids):
            sys.stdout.write('\rpid: {} traj: {} / {}'.format(pid, i, nids))
            traj = simulate(
                env, 
                policy, 
                max_steps=max_steps,
                env_kwargs=dict(egoid=egoid)
            )
            trajdict[egoid].append(traj)

    return trajdict

def parallel_collect_trajectories(
        args,
        params,
        egoids,
        n_proc,
        env_fn=utils.build_ngsim_env,
        max_steps=200,
        use_hgail=False):
    
    # build manager and dictionary mapping ego ids to list of trajectories
    manager = mp.Manager()
    trajdict = manager.dict()

    # initialize each id to an empty list
    for egoid in egoids:
        trajdict[egoid] = []

    # set policy function
    policy_fn = utils.build_hierarchy if use_hgail else utils.build_policy
    
    # partition egoids 
    proc_egoids = utils.partition_list(egoids, n_proc)

    # pool of processes, each with a set of ego ids
    pool = mp.Pool(processes=n_proc)

    # run collection
    results = []
    for pid in range(n_proc):
        res = pool.apply_async(
            collect_trajectories,
            args=(
                args, 
                params, 
                proc_egoids[pid], 
                trajdict, 
                pid,
                env_fn,
                policy_fn,
                max_steps,
                use_hgail
            )
        )
        results.append(res)

    # wait for the processes to finish
    [res.get() for res in results]

    return trajdict

def collect(
        egoids,
        args,
        exp_dir,
        use_hgail,
        params_filename,
        n_proc,
        max_steps=200):
    # load information relevant to the experiment
    params_filepath = os.path.join(exp_dir, 'imitate/log/{}'.format(params_filename))
    params = hgail.misc.utils.load_params(params_filepath)

    # validation setup 
    validation_dir = os.path.join(exp_dir, 'imitate', 'validation')
    utils.maybe_mkdir(validation_dir)
    output_filepath = os.path.join(validation_dir, '{}_trajectories.npz'.format(
        args.ngsim_filename.split('.')[0]))

    with ContextTimer():
        trajs = parallel_collect_trajectories(
            args, 
            params, 
            egoids, 
            n_proc,
            max_steps=max_steps,
            use_hgail=use_hgail
        )

    write_trajectories(output_filepath, trajs)

def load_egoids(fn, args, n_runs_per_ego_id=1, env_fn=utils.build_ngsim_env):
    offset = args.env_H + args.env_primesteps
    basedir = os.path.expanduser('~/.julia/v0.6/NGSIM/data/')
    ids_filename = fn.replace('.txt', '-index-{}-ids.h5'.format(offset))
    ids_filepath = os.path.join(basedir, ids_filename)
    if not os.path.exists(ids_filepath):
        # this should create the ids file
        env_fn(args)
        if not os.path.exists(ids_filepath):
            raise ValueError('file unable to be created, check args')
    ids = np.array(h5py.File(ids_filepath, 'r')['ids'].value)
    ids = np.tile(ids, n_runs_per_ego_id)
    return ids

if __name__ == '__main__':
    args = parser.parse_args()
    n_runs_per_ego_id = 1

    exp_dir = '../../data/experiments/NGSIM-hgail/'
    params_filename = 'itr_6.npz'
    use_hgail = True

    args_filepath = os.path.join(exp_dir, 'imitate/log/args.npz')
    args = np.load(args_filepath)['args'].item()
    filenames = [
        "trajdata_i80_trajectories-0400-0415.txt",
        "trajdata_i80_trajectories-0500-0515.txt",
        "trajdata_i80_trajectories-0515-0530.txt",
        "trajdata_i101_trajectories-0805am-0820am.txt",
        "trajdata_i101_trajectories-0820am-0835am.txt",
        "trajdata_i101_trajectories-0750am-0805am.txt"
    ]
    for fn in filenames:
        args.ngsim_filename = fn
        egoids = load_egoids(fn, args, n_runs_per_ego_id)
        collect(
            egoids,
            args,
            exp_dir=exp_dir,
            params_filename=params_filename,
            use_hgail=use_hgail,
            n_proc=args.nproc
        )
