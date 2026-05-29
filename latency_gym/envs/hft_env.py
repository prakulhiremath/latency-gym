"""Gymnasium HFT Latency Optimization Environment"""

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

import numpy as np
from typing import Any, Dict, Optional, Tuple

try:
    from latency_gym._latency_gym import LatencySimulator
except ImportError:
    raise ImportError("Failed to import compiled C++ module. Please install with: pip install -e .")


class HFTLatencyEnv(gym.Env):
    """High-Frequency Trading Latency Optimization Environment"""
    
    metadata = {"render_modes": [], "render_fps": None}

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        """Initialize HFT Latency environment"""
        super().__init__()
        
        self.seed_val = seed
        self._simulator = LatencySimulator(seed)
        
        self.action_space = spaces.MultiDiscrete([
            LatencySimulator.MAX_BATCH_SIZE - LatencySimulator.MIN_BATCH_SIZE + 1,
            LatencySimulator.MAX_POLLING_RATE - LatencySimulator.MIN_POLLING_RATE + 1,
            LatencySimulator.MAX_PREALLOC_POOL - LatencySimulator.MIN_PREALLOC_POOL + 1,
        ])
        
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([4096.0, 1e9, 1e18, 1e9], dtype=np.float32),
            dtype=np.float32
        )
        
        self._step_count = 0
        self._episode_returns = 0.0
        self._last_obs = None
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Reset the environment"""
        if seed is not None:
            self.seed_val = seed
            self._simulator = LatencySimulator(seed)
        else:
            self._simulator.reset()
        
        self._step_count = 0
        self._episode_returns = 0.0
        
        self._last_obs = self._extract_observation()
        
        return self._last_obs, {}
    
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """Execute one step of the environment"""
        action = np.asarray(action, dtype=np.int32)
        
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        
        batch_size = int(action[0]) + LatencySimulator.MIN_BATCH_SIZE
        polling_rate = int(action[1]) + LatencySimulator.MIN_POLLING_RATE
        prealloc_pool = int(action[2]) + LatencySimulator.MIN_PREALLOC_POOL
        
        batch_size = np.clip(batch_size, LatencySimulator.MIN_BATCH_SIZE, LatencySimulator.MAX_BATCH_SIZE)
        polling_rate = np.clip(polling_rate, LatencySimulator.MIN_POLLING_RATE, LatencySimulator.MAX_POLLING_RATE)
        prealloc_pool = np.clip(prealloc_pool, LatencySimulator.MIN_PREALLOC_POOL, LatencySimulator.MAX_PREALLOC_POOL)
        
        self._simulator.set_action(batch_size, polling_rate, prealloc_pool)
        self._simulator.step()
        
        obs = self._extract_observation()
        self._last_obs = obs
        
        reward = self._simulator.compute_reward(alpha=1.0, beta=0.5, gamma=2.0)
        
        self._episode_returns += reward
        self._step_count += 1
        
        terminated = False
        truncated = False
        
        info = {
            "step_count": self._step_count,
            "matched_orders": self._simulator.get_matched_orders(),
            "total_overflow": self._simulator.get_total_overflow(),
            "episode_return": self._episode_returns,
            "batch_size": batch_size,
            "polling_rate": polling_rate,
            "prealloc_pool": prealloc_pool,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _extract_observation(self) -> np.ndarray:
        """Extract observation from simulator state"""
        state = self._simulator.get_state()
        
        queue_depth = float(state.queue_depth)
        last_latency = float(state.last_latency_ns)
        latency_variance = float(state.latency_variance)
        packet_drops = float(state.packet_drops)
        
        queue_depth = np.clip(queue_depth, 0.0, 4096.0)
        last_latency = np.clip(last_latency, 0.0, 1e9)
        latency_variance = np.clip(latency_variance, 0.0, 1e18)
        packet_drops = np.clip(packet_drops, 0.0, 1e9)
        
        obs = np.array([
            queue_depth,
            last_latency,
            latency_variance,
            packet_drops
        ], dtype=np.float32)
        
        return obs
    
    def render(self) -> None:
        """Rendering not implemented"""
        pass
    
    def close(self) -> None:
        """Close environment"""
        pass
    
    def seed(self, seed: Optional[int] = None) -> list:
        """Set random seed"""
        if seed is not None:
            self.seed_val = seed
        return [self.seed_val]
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get detailed state information"""
        state = self._simulator.get_state()
        return {
            "queue_depth": float(state.queue_depth),
            "queue_fill_ratio": float(state.queue_fill_ratio),
            "last_latency_ns": int(state.last_latency_ns),
            "mean_latency_ns": int(state.mean_latency_ns),
            "p99_latency_ns": int(state.p99_latency_ns),
            "p999_latency_ns": int(state.p999_latency_ns),
            "latency_variance": float(state.latency_variance),
            "packet_drops": int(state.packet_drops),
            "step_count": int(self._simulator.get_step_count()),
            "matched_orders": int(self._simulator.get_matched_orders()),
            "total_overflow": int(self._simulator.get_total_overflow()),
        }


__all__ = ["HFTLatencyEnv"]
