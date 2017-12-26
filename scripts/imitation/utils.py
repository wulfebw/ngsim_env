
import h5py
import numpy as np
import os
import tensorflow as tf

from rllab.envs.normalized_env import normalize as normalize_env
import rllab.misc.logger as logger

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

from hgail.critic.critic import WassersteinCritic
from hgail.misc.datasets import CriticDataset, RecognitionDataset
from hgail.policies.gaussian_latent_var_mlp_policy import GaussianLatentVarMLPPolicy
from hgail.policies.latent_sampler import UniformlyRandomLatentSampler
from hgail.core.models import ObservationActionMLP
from hgail.recognition.recognition_model import RecognitionModel
from hgail.policies.scheduling import ConstantIntervalScheduler

from hgail.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
import hgail.misc.utils

from julia_env.julia_env import JuliaEnv

'''
Common 
'''
def maybe_mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

'''
Component build functions
'''

def build_ngsim_env(args, exp_dir='tmp', alpha=0.001):
    basedir = os.path.expanduser('~/.julia/v0.6/NGSIM/data')
    filepaths = [os.path.join(basedir, args.ngsim_filename)]
    env_params = dict(
        trajectory_filepaths=filepaths,
        H=args.env_H,
        primesteps=args.env_primesteps,
        terminate_on_collision=False,
        terminate_on_off_road=False,
        render_params=dict(
            viz_dir=os.path.join(exp_dir, 'viz'),
            zoom=5.
        )
    )
    env = JuliaEnv(
        env_id='NGSIMEnv',
        env_params=env_params,
        using='AutoEnvs'
    )
    # get low and high values for normalizing _real_ actions
    low, high = env.action_space.low, env.action_space.high
    env = TfEnv(normalize_env(env, normalize_obs=True, obs_alpha=alpha))
    return env, low, high

def build_critic(args, data, env, writer=None):
    if args.use_critic_replay_memory:
        critic_replay_memory = hgail.misc.utils.KeyValueReplayMemory(maxsize=3 * args.batch_size)
    else:
        critic_replay_memory = None

    critic_dataset = CriticDataset(
        data, 
        replay_memory=critic_replay_memory,
        batch_size=args.critic_batch_size
    )

    critic_network = ObservationActionMLP(
        name='critic', 
        hidden_layer_dims=args.critic_hidden_layer_dims,
        dropout_keep_prob=args.critic_dropout_keep_prob
    )
    critic = WassersteinCritic(
        obs_dim=env.observation_space.flat_dim,
        act_dim=env.action_space.flat_dim,
        dataset=critic_dataset, 
        network=critic_network,
        gradient_penalty=args.gradient_penalty,
        optimizer=tf.train.RMSPropOptimizer(args.critic_learning_rate),
        n_train_epochs=args.n_critic_train_epochs,
        summary_writer=writer,
        grad_norm_rescale=args.critic_grad_rescale,
        verbose=2,
    )
    return critic

def build_policy(args, env):
    if args.use_infogail:
        latent_sampler = UniformlyRandomLatentSampler(
            scheduler=ConstantIntervalScheduler(k=args.scheduler_k),
            name='latent_sampler',
            dim=args.latent_dim
        )
        policy = GaussianLatentVarMLPPolicy(
            name="policy",
            latent_sampler=latent_sampler,
            env_spec=env.spec,
            hidden_sizes=args.policy_mean_hidden_layer_dims,
            std_hidden_sizes=args.policy_std_hidden_layer_dims
        )
    else:
        policy = GaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_sizes=args.policy_mean_hidden_layer_dims,
            std_hidden_sizes=args.policy_std_hidden_layer_dims,
            adaptive_std=True,
            output_nonlinearity=None,
            learn_std=True
        )
    return policy

def build_recognition_model(args, env, writer=None):
    if args.use_infogail:
        recognition_dataset = RecognitionDataset(args.batch_size)
        recognition_network = ObservationActionMLP(
            name='recog', 
            hidden_layer_dims=args.recognition_hidden_layer_dims,
            output_dim=args.latent_dim
        )
        recognition_model = RecognitionModel(
            obs_dim=env.observation_space.flat_dim,
            act_dim=env.action_space.flat_dim,
            dataset=recognition_dataset, 
            network=recognition_network,
            variable_type='categorical',
            latent_dim=args.latent_dim,
            optimizer=tf.train.AdamOptimizer(args.recognition_learning_rate),
            n_train_epochs=args.n_recognition_train_epochs,
            summary_writer=writer,
            verbose=2
        )
    else:
        recognition_model = None
    return recognition_model

def build_baseline(args, env):
    return GaussianMLPBaseline(env_spec=env.spec)

def build_reward_handler(args, writer=None):
    reward_handler = hgail.misc.utils.RewardHandler(
        use_env_rewards=False,
        max_epochs=args.reward_handler_max_epochs, # epoch at which final scales are used
        critic_final_scale=1.,
        recognition_initial_scale=0.,
        recognition_final_scale=args.reward_handler_recognition_final_scale,
        summary_writer=writer,
        normalize_rewards=True,
        critic_clip_low=-100,
        critic_clip_high=100,
    )
    return reward_handler

'''
setup
'''

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
    log_filepath = os.path.join(log_dir, 'log.txt')
    logger.add_text_output(log_filepath)
    return exp_dir

'''
data utilities
'''

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

def normalize_range(x, low, high):
    low = np.array(low)
    high = np.array(high)
    mean = (high + low) / 2.
    half_range = (high - low) / 2.
    x = (x - mean) / half_range
    x = np.clip(x, -1, 1)
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
        shuffle=False,
        act_low=-1,
        act_high=1):
    
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
    # normalize actions to between -1 and 1
    act = normalize_range(act, act_low, act_high)

    return dict(
        observations=obs,
        actions=act,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )
