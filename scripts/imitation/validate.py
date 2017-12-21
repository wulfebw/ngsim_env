
import numpy as np
import os 
import tensorflow as tf

import hgail.misc.simulation

import utils

def validate_performance():
    # load information relevant to the experiment
    exp_dir = '../../data/experiments/NGSIM-v4/'
    args_filepath = os.path.join(exp_dir, 'imitate/log/args.npz')
    args = np.load(args_filepath)['args'].item()
    params_filepath = os.path.join(exp_dir, 'imitate/log/itr_6.npz')
    params = hgail.misc.utils.load_params(params_filepath)

    # optionally replace args.ngsim_filename with a different time period
    # rather than randomly iterate through trajectories, I should be able to 
    # go through all of them in a controllable manner. Todo I guess

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
            n_traj=2,
            env=env,
            policy=policy,
            max_steps=1000 # overshoot and let it terminate
        )

        print(trajs)

if __name__ == '__main__':
    validate_performance()
