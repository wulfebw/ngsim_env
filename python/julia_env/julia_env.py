
import gym
import julia
import os

from julia_env.utils import build_space

class JuliaEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, env_id, env_params, using):

        # initialize julia
        self.j = julia.Julia()
        self.j.eval('include(\"{}\")'.format(os.path.expanduser('~/.juliarc.jl')))
        self.j.using(using)

        # initialize environment
        self.env = self.j.make(env_id, env_params)
        self.observation_space = build_space(*self.j.observation_space_spec(self.env))
        self.action_space = build_space(*self.j.action_space_spec(self.env))

    def _reset(self):
        return self.j.reset(self.env)

    def _step(self, action):
        return self.j.step(self.env, action)

    def _render(self, mode='human', close=False):
        return self.j.render(self.env)
        
    def obs_names(self):
        return self.j.obs_names(self.env)