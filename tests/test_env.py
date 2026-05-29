"""Comprehensive test suite for latency-gym environment"""

import pytest
import numpy as np
import gymnasium as gym

from latency_gym.envs.hft_env import HFTLatencyEnv

try:
    from latency_gym._latency_gym import LatencySimulator
    HAS_CPP_MODULE = True
except ImportError:
    HAS_CPP_MODULE = False


class TestEnvironmentInitialization:
    """Test environment creation and initialization"""
    
    def test_env_creation(self):
        """Test basic environment creation"""
        env = HFTLatencyEnv(seed=42)
        assert env is not None
        assert isinstance(env, gym.Env)
    
    def test_action_space_shape(self):
        """Test action space has correct shape"""
        env = HFTLatencyEnv()
        assert env.action_space.shape == (3,)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
    
    def test_observation_space_shape(self):
        """Test observation space has correct shape"""
        env = HFTLatencyEnv()
        assert env.observation_space.shape == (4,)
        assert isinstance(env.observation_space, gym.spaces.Box)
    
    def test_observation_space_dtype(self):
        """Test observation space dtype is float32"""
        env = HFTLatencyEnv()
        assert env.observation_space.dtype == np.float32


class TestEnvironmentReset:
    """Test environment reset functionality"""
    
    def test_reset_returns_observation(self):
        """Test reset returns observation"""
        env = HFTLatencyEnv()
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
    
    def test_reset_observation_shape(self):
        """Test reset observation has correct shape"""
        env = HFTLatencyEnv()
        obs, info = env.reset()
        assert obs.shape == (4,)
    
    def test_reset_observation_bounds(self):
        """Test reset observation is within bounds"""
        env = HFTLatencyEnv()
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
    
    def test_reset_returns_info(self):
        """Test reset returns info dictionary"""
        env = HFTLatencyEnv()
        obs, info = env.reset()
        assert isinstance(info, dict)
    
    def test_reset_with_seed(self):
        """Test reset with explicit seed"""
        env = HFTLatencyEnv()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)


class TestEnvironmentStep:
    """Test environment step functionality"""
    
    def test_step_returns_tuple(self):
        """Test step returns 5-tuple"""
        env = HFTLatencyEnv()
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert isinstance(result, tuple)
        assert len(result) == 5
    
    def test_step_observation_shape(self):
        """Test step returns observation with correct shape"""
        env = HFTLatencyEnv()
        env.reset()
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert obs.shape == (4,)
        assert obs.dtype == np.float32
    
    def test_step_observation_in_space(self):
        """Test step observation is in observation space"""
        env = HFTLatencyEnv()
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            assert env.observation_space.contains(obs)
    
    def test_step_reward_is_float(self):
        """Test step returns float reward"""
        env = HFTLatencyEnv()
        env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert isinstance(reward, (float, np.floating))
    
    def test_step_invalid_action_raises(self):
        """Test step with invalid action raises ValueError"""
        env = HFTLatencyEnv()
        env.reset()
        invalid_action = np.array([100, 100, 100])
        with pytest.raises(ValueError):
            env.step(invalid_action)
    
    def test_step_sequence(self):
        """Test a sequence of steps"""
        env = HFTLatencyEnv()
        obs, _ = env.reset()
        
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert obs.shape == (4,)
            assert isinstance(reward, (float, np.floating))
            assert isinstance(terminated, (bool, np.bool_))
            assert isinstance(truncated, (bool, np.bool_))
            assert isinstance(info, dict)


class TestRewardFunction:
    """Test reward function properties"""
    
    def test_reward_is_negative(self):
        """Test that rewards are generally negative (penalties)"""
        env = HFTLatencyEnv()
        env.reset()
        
        negative_count = 0
        for _ in range(20):
            action = env.action_space.sample()
            _, reward, _, _, _ = env.step(action)
            if reward < 0:
                negative_count += 1
        
        assert negative_count > 10
    
    def test_reward_finite(self):
        """Test rewards are finite (not inf or nan)"""
        env = HFTLatencyEnv()
        env.reset()
        
        for _ in range(100):
            action = env.action_space.sample()
            _, reward, _, _, _ = env.step(action)
            assert np.isfinite(reward)


class TestRandomAgent:
    """Test environment with random agent"""
    
    def test_random_agent_episode(self):
        """Test running full episode with random agent"""
        env = HFTLatencyEnv(seed=42)
        obs, _ = env.reset()
        
        episode_rewards = []
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
            
            if terminated or truncated:
                break
        
        assert len(episode_rewards) > 0
        assert all(np.isfinite(r) for r in episode_rewards)
    
    def test_multiple_episodes_stability(self):
        """Test environment stability across multiple episodes"""
        env = HFTLatencyEnv()
        
        all_rewards = []
        for episode in range(5):
            obs, _ = env.reset(seed=42 + episode)
            
            for _ in range(50):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                all_rewards.append(reward)
        
        assert all(np.isfinite(r) for r in all_rewards)
        reward_std = np.std(all_rewards)
        assert reward_std > 0


class TestMemorySafety:
    """Test for memory leaks and safety"""
    
    def test_no_segfault_on_long_run(self):
        """Test environment doesn't segfault on long run"""
        env = HFTLatencyEnv()
        obs, _ = env.reset()
        
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
    
    def test_multiple_resets_safe(self):
        """Test multiple resets don't cause issues"""
        env = HFTLatencyEnv()
        
        for i in range(20):
            obs, _ = env.reset(seed=i)
            assert obs is not None
            
            for _ in range(10):
                action = env.action_space.sample()
                obs, _, _, _, _ = env.step(action)
    
    def test_env_cleanup(self):
        """Test environment can be cleaned up"""
        env = HFTLatencyEnv()
        env.reset()
        for _ in range(10):
            env.step(env.action_space.sample())
        env.close()


class TestNumericalStability:
    """Test numerical stability and edge cases"""
    
    def test_observation_no_nan(self):
        """Test observations never contain NaN"""
        env = HFTLatencyEnv()
        env.reset()
        
        for _ in range(100):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            assert not np.any(np.isnan(obs))
    
    def test_observation_no_inf(self):
        """Test observations never contain Inf"""
        env = HFTLatencyEnv()
        env.reset()
        
        for _ in range(100):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            assert not np.any(np.isinf(obs))
    
    def test_large_sequence_stability(self):
        """Test environment stability over many steps"""
        env = HFTLatencyEnv()
        env.reset()
        
        all_obs = []
        all_rewards = []
        
        for i in range(500):
            action = env.action_space.sample()
            obs, reward, _, _, _ = env.step(action)
            all_obs.append(obs)
            all_rewards.append(reward)
        
        assert all(np.all(np.isfinite(o)) for o in all_obs)
        assert all(np.isfinite(r) for r in all_rewards)


class TestStateDict:
    """Test state dictionary functionality"""
    
    def test_get_state_dict(self):
        """Test get_state_dict returns valid dictionary"""
        env = HFTLatencyEnv()
        env.reset()
        
        for _ in range(5):
            env.step(env.action_space.sample())
        
        state = env.get_state_dict()
        assert isinstance(state, dict)
        assert "queue_depth" in state
        assert "latency_variance" in state
        assert "packet_drops" in state
    
    def test_state_dict_values_numeric(self):
        """Test state dict values are numeric"""
        env = HFTLatencyEnv()
        env.reset()
        env.step(env.action_space.sample())
        
        state = env.get_state_dict()
        for key, value in state.items():
            assert isinstance(value, (int, float, np.number))


@pytest.mark.skipif(not HAS_CPP_MODULE, reason="C++ module not compiled")
class TestCppSimulator:
    """Test C++ simulator directly"""
    
    def test_simulator_creation(self):
        """Test LatencySimulator creation"""
        sim = LatencySimulator(seed=42)
        assert sim is not None
    
    def test_simulator_reset(self):
        """Test simulator reset"""
        sim = LatencySimulator()
        sim.reset()
        assert sim.get_step_count() == 0
    
    def test_simulator_step(self):
        """Test simulator step"""
        sim = LatencySimulator()
        sim.reset()
        sim.set_action(4, 5, 2)
        sim.step()
        assert sim.get_step_count() == 1
    
    def test_simulator_state(self):
        """Test getting simulator state"""
        sim = LatencySimulator()
        sim.reset()
        sim.step()
        state = sim.get_state()
        assert state is not None
    
    def test_simulator_reward(self):
        """Test reward computation"""
        sim = LatencySimulator()
        sim.reset()
        sim.step()
        reward = sim.compute_reward()
        assert np.isfinite(reward)


class TestGymIntegration:
    """Test Gymnasium integration"""
    
    def test_make_env(self):
        """Test creating env via gym.make"""
        env = gym.make("hft-latency-v0")
        assert env is not None
        assert isinstance(env, gym.Env)
    
    def test_make_env_reset_step(self):
        """Test gym.make env reset and step"""
        env = gym.make("hft-latency-v0")
        obs, info = env.reset()
        assert obs is not None
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        assert np.isfinite(reward)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
