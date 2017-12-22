
import numpy as np
import os
import tensorflow as tf

import auto_validator
import hyperparams
import utils

from hgail.algos.gail import GAIL

# setup
exp_name = "NGSIM-v4"
exp_dir = utils.set_up_experiment(exp_name=exp_name, phase='imitate')
saver_dir = os.path.join(exp_dir, 'imitate', 'log')
saver_filepath = os.path.join(saver_dir, 'checkpoint')
args = hyperparams.parse_args()
np.savez(os.path.join(saver_dir, 'args'), args=args)
summary_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'imitate', 'summaries'))

# build components
env, act_low, act_high = utils.build_ngsim_env(args)
data = utils.load_data(args.expert_filepath, act_low=act_low, act_high=act_high)
critic = utils.build_critic(args, data, env, summary_writer)
policy = utils.build_policy(args, env)
recognition_model = utils.build_recognition_model(args, env, summary_writer)
baseline = utils.build_baseline(args, env)
reward_handler = utils.build_reward_handler(args, summary_writer)
validator = auto_validator.AutoValidator(summary_writer, data['obs_mean'], data['obs_std'])

# build algo 
saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=.5)
algo = GAIL(
    critic=critic,
    recognition=recognition_model,
    reward_handler=reward_handler,
    env=env,
    policy=policy,
    baseline=baseline,
    validator=validator,
    batch_size=args.batch_size,
    max_path_length=args.max_path_length,
    n_itr=args.n_itr,
    discount=args.discount,
    step_size=args.trpo_step_size,
    saver=saver,
    saver_filepath=saver_filepath,
    force_batch_sampler=True,
    snapshot_env=False,
    plot=True,
    optimizer_args=dict(
        max_backtracks=50
    )
)

# run it
with tf.Session() as session:
    
    # running the initialization here to allow for later loading
    # NOTE: rllab batchpoplot runs this before training as well 
    # this means that any loading subsequent to this is nullified 
    # you have to comment of that initialization for any loading to work
    session.run(tf.global_variables_initializer())

    # loading
    if args.params_filepath != '':
        algo.load(args.params_filepath)

    # run training
    algo.train(sess=session)
