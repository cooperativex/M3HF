#!/usr/bin/env python3
"""
Environment Demo Script

This script demonstrates how to interact with the Overcooked multi-agent
environment, showing observation spaces, action spaces, and basic gameplay.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from gym_macro_overcooked import gym_macro_overcooked
import pygame

def demo_environment_registration():
    """Demo environment registration and basic info."""
    print("=== Environment Registration Demo ===")
    
    # Register environments
    try:
        import gym_macro_overcooked
        print("✓ gym_macro_overcooked imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import gym_macro_overcooked: {e}")
        return
    
    # List available environments
    available_envs = ['Overcooked-v1', 'Overcooked-MA-v1']
    print(f"Available environments: {available_envs}")

def demo_environment_creation():
    """Demo environment creation with different configurations."""
    print("\n=== Environment Creation Demo ===")
    
    configs = [
        {'env_id': 'Overcooked-MA-v1', 'n_agent': 2, 'map_type': 'A', 'task': 3},
        {'env_id': 'Overcooked-MA-v1', 'n_agent': 3, 'map_type': 'B', 'task': 6},
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config}")
        try:
            env = gym.make(
                config['env_id'],
                grid_dim=[7, 7],
                map_type=config['map_type'],
                n_agent=config['n_agent'],
                obs_radius=2,
                task=config['task'],
                mode='vector',
                debug=False
            )
            
            print(f"✓ Environment created successfully")
            print(f"  Observation space: {env.observation_space}")
            print(f"  Action space: {env.action_space}")
            
            # Get sample observation
            obs, info = env.reset()
            print(f"  Sample observation shape: {[ob.shape for ob in obs]}")
            print(f"  Number of agents: {len(obs)}")
            
            env.close()
            
        except Exception as e:
            print(f"✗ Failed to create environment: {e}")

def demo_observation_space():
    """Demo observation space analysis."""
    print("\n=== Observation Space Demo ===")
    
    env = gym.make(
        'Overcooked-MA-v1',
        grid_dim=[7, 7],
        map_type='A',
        n_agent=3,
        obs_radius=2,
        task=6,  # Most complex task
        mode='vector',
        debug=False
    )
    
    obs, info = env.reset()
    
    print(f"Number of agents: {len(obs)}")
    for i, agent_obs in enumerate(obs):
        print(f"Agent {i} observation shape: {agent_obs.shape}")
        print(f"Agent {i} observation: {agent_obs[:10]}...")  # Show first 10 elements
    
    # Explain observation structure
    print("\nObservation structure (32 elements):")
    print("  [0:3]   - Tomato position and status")
    print("  [3:6]   - Lettuce position and status") 
    print("  [6:9]   - Onion position and status")
    print("  [9:11]  - Plate 1 position")
    print("  [11:13] - Plate 2 position")
    print("  [13:15] - Knife 1 position")
    print("  [15:17] - Knife 2 position")
    print("  [17:19] - Delivery counter position")
    print("  [19:21] - Agent 1 position")
    print("  [21:23] - Agent 2 position")
    print("  [23:25] - Agent 3 position")
    print("  [25:32] - Task one-hot encoding")
    
    env.close()

def demo_action_space():
    """Demo action space and available actions."""
    print("\n=== Action Space Demo ===")
    
    env = gym.make(
        'Overcooked-MA-v1',
        grid_dim=[7, 7],
        map_type='A',
        n_agent=3,
        obs_radius=0,
        task=3,
        mode='vector',
        debug=False
    )
    
    print(f"Action space: {env.action_space}")
    
    # Available macro actions
    macro_actions = {
        0: "stay",
        1: "get_tomato", 
        2: "get_lettuce",
        3: "get_onion",
        4: "get_plate1",
        5: "get_plate2", 
        6: "go_knife1",
        7: "go_knife2",
        8: "deliver",
        9: "chop",
        10: "right",
        11: "down", 
        12: "left",
        13: "up"
    }
    
    print("Available macro actions:")
    for action_id, action_name in macro_actions.items():
        print(f"  {action_id}: {action_name}")
    
    env.close()

def demo_simple_episode():
    """Demo a simple episode with random actions."""
    print("\n=== Simple Episode Demo ===")
    
    env = gym.make(
        'Overcooked-MA-v1',
        grid_dim=[7, 7],
        map_type='A',
        n_agent=3,
        obs_radius=2,
        task=3,  # Lettuce-tomato salad
        mode='vector',
        debug=False
    )
    
    obs, info = env.reset()
    print("Episode started")
    
    total_rewards = [0] * len(obs)
    episode_steps = 0
    max_steps = 50  # Limit for demo
    
    for step in range(max_steps):
        # Take random actions for each agent
        actions = [env.action_space.sample() for _ in range(len(obs))]
        
        # Step environment
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Update totals
        for i, reward in enumerate(rewards):
            total_rewards[i] += reward
        episode_steps += 1
        
        # Print step info
        if step % 10 == 0:
            print(f"Step {step}: rewards={rewards}, terminated={terminated}")
        
        # Check if episode ended
        if any(terminated) or any(truncated):
            break
    
    print(f"Episode ended after {episode_steps} steps")
    print(f"Total rewards per agent: {total_rewards}")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    
    env.close()

def demo_task_types():
    """Demo different task types."""
    print("\n=== Task Types Demo ===")
    
    task_list = [
        "tomato salad",           # task 0
        "lettuce salad",          # task 1
        "onion salad",            # task 2
        "lettuce-tomato salad",   # task 3
        "onion-tomato salad",     # task 4
        "lettuce-onion salad",    # task 5
        "lettuce-onion-tomato salad"  # task 6
    ]
    
    print("Available tasks:")
    for task_id, task_name in enumerate(task_list):
        print(f"  Task {task_id}: {task_name}")
    
    # Demo creating environments for different tasks
    for task_id in [0, 3, 6]:  # Simple, medium, complex
        print(f"\nTesting Task {task_id} ({task_list[task_id]}):")
        
        env = gym.make(
            'Overcooked-MA-v1',
            grid_dim=[7, 7],
            map_type='A',
            n_agent=3,
            obs_radius=2,
            task=task_id,
            mode='vector',
            debug=False
        )
        
        obs, info = env.reset()
        print(f"  Task one-hot encoding: {obs[0][25:32]}")
        
        env.close()

def demo_partial_observability():
    """Demo partial observability with different observation radii."""
    print("\n=== Partial Observability Demo ===")
    
    obs_radii = [0, 1, 2, 3]  # 0 = full observability
    
    for radius in obs_radii:
        print(f"\nObservation radius: {radius}")
        
        env = gym.make(
            'Overcooked-MA-v1',
            grid_dim=[7, 7],
            map_type='A',
            n_agent=3,
            obs_radius=radius,
            task=3,
            mode='vector',
            debug=False
        )
        
        obs, info = env.reset()
        
        if radius == 0:
            print("  Full observability - agents can see everything")
        else:
            observable_area = (2 * radius + 1) ** 2
            print(f"  Partial observability - each agent sees {observable_area} cells")
        
        # Show how observation changes
        agent_0_obs = obs[0]
        print(f"  Agent 0 observation sample: {agent_0_obs[:5]}...")
        
        env.close()

def main():
    """Run all demos."""
    print("Overcooked Multi-Agent Environment Demo")
    print("=" * 50)
    
    # Run all demos
    demo_environment_registration()
    demo_environment_creation()
    demo_observation_space()
    demo_action_space()
    demo_simple_episode()
    demo_task_types()
    demo_partial_observability()
    
    print("\n" + "=" * 50)
    print("Demo completed! Try running:")
    print("  python play.py --env_id Overcooked-MA-v1 --n_agent 3 --task 6")
    print("to play the environment manually.")

if __name__ == '__main__':
    main()