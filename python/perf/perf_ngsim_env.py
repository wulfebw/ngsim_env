
import numpy as np
import os

from julia_env.julia_env import JuliaEnv

from context_timer import ContextTimer

def perf_ngsim_env_step():
    basedir = os.path.expanduser('~/.julia/v0.6/NGSIM/data')
    filename = 'trajdata_i101_trajectories-0750am-0805am.txt'
    filepaths = [os.path.join(basedir, filename)]
    env = JuliaEnv(
        env_id='NGSIMEnv',
        env_params=dict(
            trajectory_filepaths=filepaths,
        ),
        using='AutoEnvs'
    )
    n_steps = 20000
    action = np.array([1.,0.])
    env.reset()
    with ContextTimer():
        for _ in range(n_steps):
            _, _, terminal, _ = env.step(action)
            if terminal:
                env.reset()

if __name__ == '__main__':
    perf_ngsim_env_step()