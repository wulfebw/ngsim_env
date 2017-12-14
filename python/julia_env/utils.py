
import gym
from gym import spaces
import numpy as np

def build_space(shape, space_type, info={}):
    if space_type == 'Box':
        if 'low' in info and 'high' in info:
            low = info['low']
            high = info['high']
            msg = 'shape = {}\tlow.shape = {}\thigh.shape={}'.format(
                shape, low.shape, high.shape)
            assert shape == np.shape(low) and shape == np.shape(high), msg
            return spaces.Box(low=low, high=high)
        else:
            return spaces.Box(low=-np.inf, high=np.inf, shape=shape)        
    elif space_type == 'Discrete':
        assert len(shape) == 1, 'invalid shape for Discrete space'
        return spaces.Discrete(shape)
    else:
        raise(ValueError('space type not implemented: {}'.format(space_type)))