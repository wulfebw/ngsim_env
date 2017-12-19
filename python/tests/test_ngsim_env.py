
import numpy as np
import os
import unittest

from julia_env.julia_env import JuliaEnv

class TestNGSIMEnv(unittest.TestCase):

    def test_ngsim_env(self):
        basedir = os.path.expanduser('~/.julia/v0.6/NGSIM/data')
        filename = 'trajdata_i101_trajectories-0750am-0805am.txt'
        filepaths = [os.path.join(basedir, filename)]
        env = JuliaEnv(
            env_id='NGSIMEnv',
            env_params=dict(trajectory_filepaths=filepaths),
            using='AutoEnvs'
        )
        x = env.reset()
        nx, r, t, info = env.step(np.array([0.,0.]))
        self.assertTrue(np.sum(np.abs(x-nx)) > 1e-1)

if __name__ == '__main__':
    unittest.main()