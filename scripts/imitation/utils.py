
import h5py
import numpy as np
import os

import rllab.misc.logger as logger

from julia_env.julia_env import JuliaEnv

def build_ngsim_env(filename='trajdata_i101_trajectories-0750am-0805am.txt'):
    basedir = os.path.expanduser('~/.julia/v0.5/NGSIM/data')
    filepaths = [os.path.join(basedir, filename)]
    env = JuliaEnv(
            env_id='NGSIMEnv',
            env_params=dict(trajectory_filepaths=filepaths),
            using='AutoEnvs'
        )
    return env

def maybe_mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

def latest_snapshot(exp_dir, phase='train'):
    snapshot_dir = os.path.join(exp_dir, phase, 'log')
    snapshots = glob.glob('{}/itr_*.pkl'.format(snapshot_dir))
    latest = sorted(snapshots, reverse=True)[0]
    return latest

def set_up_experiment(
        exp_name, 
        phase, 
        exp_home='../../data/experiments/',
        snapshot_gap=5):
    maybe_mkdir(exp_home)
    exp_dir = os.path.join(exp_home, exp_name)
    maybe_mkdir(exp_dir)
    phase_dir = os.path.join(exp_dir, phase)
    maybe_mkdir(phase_dir)
    log_dir = os.path.join(phase_dir, 'log')
    maybe_mkdir(log_dir)
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode('gap')
    logger.set_snapshot_gap(snapshot_gap)
    return exp_dir

def compute_lengths(arr):
    sums = np.sum(np.array(arr), axis=2)
    lengths = []
    for sample in sums:
        zero_idxs = np.where(sample == 0.)[0]
        if len(zero_idxs) == 0:
            lengths.append(len(sample))
        else:
            lengths.append(zero_idxs[0])
    return np.array(lengths)

def normalize(x):
    mean = np.mean(x, axis=0, keepdims=True)
    x = x - mean
    std = np.std(x, axis=0, keepdims=True) + 1e-8
    x = x / std
    return x, mean, std

def normalize_range(x, orig_lb, orig_ub, lb, ub):
    orig_lb = np.array(orig_lb)
    orig_ub = np.array(orig_ub)
    lb = np.array(lb)
    ub = np.array(ub)

    scale = (ub - lb) / (orig_ub - orig_lb)
    x = lb + (x * scale + 1.) * 0.5 * (ub - lb)
    x = np.clip(x, lb, ub)
    return x

def load_x_feature_names(filepath):
    f = h5py.File(filepath, 'r')
    xs = []
    for i in range(1,6+1):
        if str(i) in f.keys():
            xs.append(f[str(i)])
    x = np.concatenate(xs)
    feature_names = f.attrs['feature_names']
    return x, feature_names

def load_data(
        filepath,
        act_keys=['accel', 'turn_rate_global'],
        debug_size=None,
        min_length=40,
        normalize_data=True,
        shuffle=False):
    
    # loading varies based on dataset type
    x, feature_names = load_x_feature_names(filepath)
    
    # optionally keep it to a reasonable size
    if debug_size is not None:
        x = x[:debug_size]
       
    if shuffle:
        idxs = np.random.permutation(len(x))
        x = x[idxs]

    # compute lengths of the samples before anything else b/c this is fragile
    lengths = compute_lengths(x)

    # flatten the dataset to (n_samples, n_features)
    # taking only the valid timesteps from each sample
    # i.e., throw out timeseries information
    xs = []
    for i, l in enumerate(lengths):
        xs.append(x[i,:l])
    x = np.concatenate(xs)

    # split into observations and actions
    # redundant because the environment is not able to extract actions
    obs = x
    act_idxs = [i for (i,n) in enumerate(feature_names) if n in act_keys]
    act = x[:, act_idxs]

    # normalize it all, _no_ test / val split
    obs, obs_mean, obs_std = normalize(obs)
    # hardcoded values emulate the ngsim_env
    # should pass these in from the enviornment
    act = normalize_range(
        act, 
        orig_lb=[-2,-.05], 
        orig_ub=[2,.05],
        lb=[-1,-1],
        ub=[1,1]
    )
    
    return dict(
        observations=obs,
        actions=act,
        obs_mean=obs_mean,
        obs_std=obs_std
    )

if __name__ == '__main__':
    orig_lb = [-1,-.05]
    orig_ub = [1,.05]
    lb = [-1,-1]
    ub = [1,1]
    x = [
        [0,0],
        [-1,-.05],
        [1,.05]
    ]
    print(normalize_range(x, orig_lb, orig_ub, lb, ub))
