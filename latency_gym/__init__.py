"""Latency Gym: High-performance HFT latency simulator for Gymnasium"""

__version__ = "0.1.0"
__author__ = "Latency Gym Contributors"

import gymnasium as gym
from gymnasium.envs.registration import register

from .envs.hft_env import HFTLatencyEnv

register(
    id="hft-latency-v0",
    entry_point="latency_gym.envs:HFTLatencyEnv",
    max_episode_steps=1000,
    reward_threshold=None,
    nondeterministic=False,
    kwargs={}
)

__all__ = ["HFTLatencyEnv", "__version__"]
