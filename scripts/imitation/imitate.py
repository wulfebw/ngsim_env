
import numpy as np
import os
import tensorflow as tf

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

from hgail.critic.critic import WassersteinCritic
from hgail.misc.datasets import CriticDataset, RecognitionDataset
from hgail.policies.categorical_latent_var_mlp_policy import CategoricalLatentVarMLPPolicy
from hgail.algos.gail import GAIL
from hgail.policies.latent_sampler import UniformlyRandomLatentSampler
from hgail.core.models import ObservationActionMLP
from hgail.recognition.recognition_model import RecognitionModel
from hgail.policies.scheduling import ConstantIntervalScheduler
import hgail.misc.utils

import auto_validator
import utils

# setup
exp_name = "NGSIM-v0"
exp_dir = utils.set_up_experiment(exp_name=exp_name, phase='imitate')
saver_dir = os.path.join(exp_dir, 'imitate', 'log')
saver_filepath = os.path.join(saver_dir, 'checkpoint')

# constants
use_infogail = False
use_critic_replay_memory = True
latent_dim = 2
real_data_maxsize = None
batch_size = 8000
n_critic_train_epochs = 30
n_recognition_train_epochs = 30
scheduler_k = 20
trpo_step_size = .5
critic_learning_rate = .0005
critic_dropout_keep_prob = .8
recognition_learning_rate = .0001
initial_filepath = None

if initial_filepath is None:
    start_itr = 0
else:
    start_itr = int(initial_filepath[initial_filepath.rfind('-')+1:])
n_itr = start_itr + 1000
max_path_length = 1000

# load env
# filename = 'trajdata_i101_trajectories-0750am-0805am.txt'
filename = '2_simple.txt'
env = utils.build_ngsim_env(filename, H=65, primesteps=2)
# get low and high values for normalizing _real_ actions
low, high = env.action_space.low, env.action_space.high
env = TfEnv(normalize(env, normalize_obs=True))

# critic dataset
# build action normalizer first to use it on the actions 
# expert_data_filepath = '../../data/trajectories/ngsim.h5'
expert_data_filepath = '../../data/trajectories/2_simple.h5'
data = utils.load_data(expert_data_filepath, act_low=low, act_high=high)


if use_critic_replay_memory:
    critic_replay_memory = hgail.misc.utils.KeyValueReplayMemory(maxsize=3 *  batch_size)
else:
    critic_replay_memory = None

critic_dataset = CriticDataset(
    data, 
    replay_memory=critic_replay_memory,
    batch_size=1000
)

# session for actual training
with tf.Session() as session:
    
    # summary writer 
    summary_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'imitate', 'summaries'))

    # build the critic
    critic_network = ObservationActionMLP(
        name='critic', 
        hidden_layer_dims=[128,128,64],
        dropout_keep_prob=critic_dropout_keep_prob
    )
    critic = WassersteinCritic(
        obs_dim=env.observation_space.flat_dim,
        act_dim=env.action_space.flat_dim,
        dataset=critic_dataset, 
        network=critic_network,
        gradient_penalty=1.,
        optimizer=tf.train.RMSPropOptimizer(critic_learning_rate),
        n_train_epochs=n_critic_train_epochs,
        summary_writer=summary_writer,
        grad_norm_rescale=50.,
        verbose=2,
    )

    if use_infogail:
        # recognition model
        recognition_dataset = RecognitionDataset(batch_size)
        recognition_network = ObservationActionMLP(
            name='recog', 
            hidden_layer_dims=[32,32],
            output_dim=latent_dim
        )
        recognition_model = RecognitionModel(
            obs_dim=env.observation_space.flat_dim,
            act_dim=env.action_space.n,
            dataset=recognition_dataset, 
            network=recognition_network,
            variable_type='categorical',
            latent_dim=latent_dim,
            optimizer=tf.train.AdamOptimizer(recognition_learning_rate, beta1=.5, beta2=.9),
            n_train_epochs=n_recognition_train_epochs,
            summary_writer=summary_writer,
            verbose=2
        )

        # build the policy
        latent_sampler = UniformlyRandomLatentSampler(
            scheduler=ConstantIntervalScheduler(k=scheduler_k),
            name='latent_sampler',
            dim=latent_dim
        )
        policy = CategoricalLatentVarMLPPolicy(
            policy_name="policy",
            latent_sampler=latent_sampler,
            env_spec=env.spec,
            hidden_sizes=(64,64)
        )
    else:
        # build the policy
        policy = GaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_sizes=(128,128),
            adaptive_std=True,
            output_nonlinearity=None,
            learn_std=True
        )
        recognition_model = None

    # build gail
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    reward_handler = hgail.misc.utils.RewardHandler(
        use_env_rewards=False,
        max_epochs=50, # epoch at which final scales are used
        critic_final_scale=1.,
        recognition_initial_scale=0.
    )

    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=.5)
    if initial_filepath:
        saver.restore(session, initial_filepath)

    # validator
    validator = auto_validator.AutoValidator(summary_writer)

    algo = GAIL(
        critic=critic,
        recognition=recognition_model,
        reward_handler=reward_handler,
        env=env,
        policy=policy,
        baseline=baseline,
        validator=validator,
        batch_size=batch_size,
        max_path_length=max_path_length,
        n_itr=n_itr,
        start_itr=start_itr,
        discount=.9,
        step_size=trpo_step_size,
        saver=saver,
        saver_filepath=saver_filepath,
        force_batch_sampler=True,
        snapshot_env=False,
        plot=True,
        optimizer_args=dict(
            max_backtracks=50
        )
    )

    # run training
    algo.train(sess=session)
