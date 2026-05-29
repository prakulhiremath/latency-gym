#!/usr/bin/env python3
"""Example script demonstrating latency-gym usage."""

import gymnasium as gym
import numpy as np


def main():
    print("=" * 70)
    print("Latency Gym - Example Usage")
    print("=" * 70)
    
    print("\n[1] Creating environment...")
    env = gym.make("hft-latency-v0")
    print(f"    ✓ Action space: {env.action_space}")
    print(f"    ✓ Observation space: {env.observation_space}")
    
    print("\n[2] Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"    ✓ Initial observation shape: {obs.shape}")
    print(f"    ✓ Initial observation: {obs}")
    
    print("\n[3] Executing single step with custom action...")
    action = np.array([3, 4, 1])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"    ✓ Action: {action}")
    print(f"    ✓ Observation: {obs}")
    print(f"    ✓ Reward: {reward:.4f}")
    print(f"    ✓ Step count: {info['step_count']}")
    
    print("\n[4] Running 100-step episode with random actions...")
    env.reset(seed=42)
    total_reward = 0.0
    rewards_list = []
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards_list.append(reward)
        
        if (step + 1) % 25 == 0:
            print(f"    Step {step + 1:3d}: Reward={reward:10.2f}, Queue={obs[0]:6.1f}")
    
    print(f"    ✓ Episode total reward: {total_reward:.2f}")
    print(f"    ✓ Mean step reward: {np.mean(rewards_list):.2f}")
    
    print("\n[5] Inspecting detailed environment state...")
    env.reset(seed=42)
    
    for _ in range(50):
        env.step(env.action_space.sample())
    
    state_dict = env.get_state_dict()
    
    print(f"    Queue: {state_dict['queue_depth']:.1f}")
    print(f"    Mean latency: {state_dict['mean_latency_ns']:,.0f} ns")
    print(f"    p99 latency: {state_dict['p99_latency_ns']:,.0f} ns")
    print(f"    p99.9 latency: {state_dict['p999_latency_ns']:,.0f} ns")
    print(f"    Variance: {state_dict['latency_variance']:.2e}")
    print(f"    Packet drops: {state_dict['packet_drops']:.0f}")
    
    print("\n[6] Testing observation space compliance...")
    env.reset()
    violations = 0
    
    for _ in range(1000):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        
        if not env.observation_space.contains(obs):
            violations += 1
    
    if violations == 0:
        print(f"    ✓ All 1000 observations within valid bounds")
    else:
        print(f"    ✗ {violations} observations violated bounds")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
