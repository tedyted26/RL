from collections import deque

import gym
import numpy as np

from baselines_wrappers import VecEnvWrapper
from baselines_wrappers.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, ScaledFloatFrame, \
    ClipRewardEnv, WarpFrame
from baselines_wrappers.wrappers import TimeLimit


def make_atari_deepmind(env_id, max_episode_steps=None, scale_values=False, clip_rewards=True):
    env = gym.make(env_id)
    env = NoopResetEnv(env, noop_max=30)

    if 'NoFrameskip' in env.spec.id:
        env = MaxAndSkipEnv(env, skip=4)

    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

    env = EpisodicLifeEnv(env)

    env = WarpFrame(env)

    if scale_values:
        env = ScaledFloatFrame(env)

    if clip_rewards:
        env = ClipRewardEnv(env)

    env = TransposeImageObs(env, op=[2, 0, 1])  # Convert to torch order (C, H, W)

    return env

class TransposeImageObs(gym.ObservationWrapper):
    def __init__(self, env, op):
        super().__init__(env)
        assert len(op) == 3, "Op must have 3 dimensions"

        self.op = op

        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [
                obs_shape[self.op[0]],
                obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, obs):
        return obs.transpose(self.op[0], self.op[1], self.op[2])