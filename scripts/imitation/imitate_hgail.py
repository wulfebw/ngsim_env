
import numpy as np
import os
import tensorflow as tf

from hgail.algos.hgail_impl import HGAIL

import auto_validator
import hyperparams
import utils

# setup
args = hyperparams.parse_args()
exp_dir = utils.set_up_experiment(exp_name=args.exp_name, phase='imitate')
saver_dir = os.path.join(exp_dir, 'imitate', 'log')
saver_filepath = os.path.join(saver_dir, 'checkpoint')
np.savez(os.path.join(saver_dir, 'args'), args=args)
summary_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'imitate', 'summaries'))

# build components
env, act_low, act_high = utils.build_ngsim_env(args, exp_dir)
data = utils.load_data(args.expert_filepath, act_low=act_low, act_high=act_high)
critic = utils.build_critic(args, data, env, summary_writer)
hierarchy = utils.build_hierarchy(args, env, summary_writer)
algo = HGAIL(critic=critic, hierarchy=hierarchy)

# session for actual training
with tf.Session() as session:
 
    # running the initialization here to allow for later loading
    # NOTE: rllab batchpolopt runs this before training as well 
    # this means that any loading subsequent to this is nullified 
    # you have to comment of that initialization for any loading to work
    session.run(tf.global_variables_initializer())

    # loading
    if args.params_filepath != '':
        algo.load(args.params_filepath)

    # run training
    algo.train(sess=session)
