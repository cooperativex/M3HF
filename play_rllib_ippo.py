import argparse
import gymnasium as gym
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper
from utils import serialize_for_wandb, extract_metrics_for_wandb
import json
import wandb
import numpy as np
import os   
import multiprocessing

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
import torch.nn as nn

TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]


class OvercookedWrapper:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        
        if isinstance(obs, list):
            return {i: obs[i] for i in range(len(obs))}, {}
        elif isinstance(obs, dict):
            return obs, {}
        else:
            raise ValueError(f"Unexpected observation type from reset: {type(obs)}")

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info 


class MultiAgentOvercookedEnv(MultiAgentEnv):
    def __init__(self, env_config):
        env_id = env_config.pop('env_id')  # Remove env_id from env_config
        rewardList = {"subtask finished": 10, "correct delivery": 200, "wrong delivery": -5, "step penalty": -0.1}
        env_config['rewardList'] = rewardList
        self.env = gym.make(env_id, **env_config)
        if env_id == "Overcooked-MA-v1":
            self.env = MacEnvWrapper(self.env)
        elif env_id == "Overcooked-v1":
            self.env = OvercookedWrapper(self.env)
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.num_agents = env_config['n_agent']
        self.env_id = env_id
        self.map_type = env_config['map_type']
        print(f"Initialized MultiAgentOvercookedEnv with {self.num_agents} agents")
        print(f"Environment: {self.env_id}")
        print(f"Action Space: {self.action_space}")
        print(f"Observation Space: {self.observation_space}")
        
        self.cumulative_rewards = {i: 0 for i in range(self.num_agents)}
        self.discount_factor = 0.99
        self.step_count = 0
        self.max_steps = 200

    def reset(self, seed=None, options=None):
        self.cumulative_rewards = {i: 0 for i in range(self.num_agents)}
        self.step_count = 0
        return self.env.reset()
    
    def step(self, action_dict):
        actions = [action_dict[i] for i in range(self.num_agents)]
        obs, rewards, done, info = self.env.step(actions)
        
        self.step_count += 1
        
        if isinstance(obs, list):
            obs = {i: obs[i] for i in range(len(obs))}
        elif not isinstance(obs, dict):
            raise ValueError(f"Unexpected observation type from step: {type(obs)}")
        
        if isinstance(rewards, (int, float)):
            rewards = {i: rewards for i in range(self.num_agents)}
        elif isinstance(rewards, list):
            rewards = {i: rewards[i] for i in range(len(rewards))}
        elif not isinstance(rewards, dict):
            raise ValueError(f"Unexpected rewards type: {type(rewards)}")
        
        # Check for wrong dish delivery or max steps reached
        wrong_delivery = any(reward == self.env.rewardList["wrong delivery"] for reward in rewards.values())
        max_steps_reached = self.step_count >= self.max_steps
        
        if wrong_delivery or max_steps_reached:
            dones = {i: True for i in range(self.num_agents)}
            dones["__all__"] = True
        elif isinstance(done, bool):
            dones = {i: done for i in range(self.num_agents)}
            dones["__all__"] = done
        elif isinstance(done, dict):
            dones = done
            dones["__all__"] = all(dones.values())
        else:
            raise ValueError(f"Unexpected done type: {type(done)}")
        
        truncateds = {i: False for i in range(self.num_agents)}
        truncateds["__all__"] = False
        
        if isinstance(info, dict):
            new_info = {'__common__': info}
        else:
            new_info = {}
        
        # Update cumulative rewards
        for i in range(self.num_agents):
            self.cumulative_rewards[i] += rewards[i] * (self.discount_factor ** self.step_count)
        
        if self.env_id == "Overcooked-v1":
            return obs, self.cumulative_rewards.copy(), dones, truncateds, new_info
        else:
            return obs, rewards, dones, truncateds, new_info

    def render(self, mode='human'):
        return self.env.render(mode)

def create_env(env_config):
    return MultiAgentOvercookedEnv(env_config)

def train_multi_agent_ppo(env_config, num_workers=2, num_gpus=3, training_iterations=200, use_wandb=False, wandb_project_name="overcooked_multi_agent"):
    print("Starting train_multi_agent_ppo function")
    
    # Get the number of CPUs available
    num_cpus = multiprocessing.cpu_count()
    print(f"Number of CPUs available: {num_cpus}")
    
    # Set workers per CPU (e.g., 2 workers per CPU)
    workers_per_cpu = 1
    total_workers = int((num_cpus / workers_per_cpu) * 0.45)
    
    ray.init(ignore_reinit_error=True)
    print(f"Ray initialized with {num_cpus} CPUs and {num_gpus} GPUs")

    if use_wandb:
        import time 
        run_name = time.strftime("%Y%m%d-%H%M%S")
        wandb.init(project=wandb_project_name, group='IPPO', name= run_name, config=env_config)
        print(f"Wandb initialized with project: {wandb_project_name}")
        



    register_env("multi_agent_overcooked", lambda config: MultiAgentOvercookedEnv(config))
    print("Environment registered")


    
    temp_env = MultiAgentOvercookedEnv(env_config.copy())
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    
    
    
    config = (
        PPOConfig()
        .environment("multi_agent_overcooked", env_config=env_config)
        .framework("torch")
        .rollouts(num_rollout_workers=total_workers, num_envs_per_worker=1)
        .resources(num_gpus=num_gpus, num_gpus_per_worker= (3 / total_workers)/2, num_cpus_per_worker=workers_per_cpu)
        .multi_agent(
            policies={
                f"agent_{i}": (None, obs_space, act_space, {})
                for i in range(env_config['n_agent'])
            },
            policy_mapping_fn=lambda agent_id, episode, **kwargs: f"agent_{agent_id}",
        )
        .training(
            train_batch_size=5120,
            sgd_minibatch_size=1024,
            num_sgd_iter=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            kl_coeff=0.2,
            grad_clip=0.5,
        )
    )

    config.rollout_fragment_length = 'auto'
    config.model["fcnet_hiddens"] = (512, 512, 512)
    config.model["fcnet_activation"] = "relu"
    config.model["conv_activation"] = "relu"
    config.model["post_fcnet_hiddens"] = (512, 512)
    config.model["post_fcnet_activation"] = "relu"
    config.model["use_lstm"] = False
    config.model["lstm_use_prev_action"] = False
    config.model["lstm_use_prev_reward"] = False
    config.model["lstm_cell_size"] = 512

    print("PPO config created")

    algo = config.build()
    print("PPO algorithm built")
    
    
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory created at {checkpoint_dir}")



    for i in range(training_iterations):
        print(f"Starting iteration {i+1}/{training_iterations}")
        result = algo.train()
        
        # Access episode counts
        episodes_total = result.get('episodes_total', 0)
        episodes_this_iter = result.get('episodes_this_iter', 0)
        
        print(f"  Episodes this iteration: {episodes_this_iter}")
        print(f"  Total episodes: {episodes_total}")
        print(f"  Average reward: {result['episode_reward_mean']}")
        print(f"  Maximum reward: {result['episode_reward_max']}")
        print(f"  Minimum reward: {result['episode_reward_min']}")
        print(f"  Average episode length: {result['episode_len_mean']}")
        
        if use_wandb:
            metrics = extract_metrics_for_wandb(result)
            wandb.log(metrics)
        
        # Save checkpoint every 100 iterations
        if (i + 1) % 100 == 0:
            checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{i+1}")
            algo.save(checkpoint_file)
            print(f"Checkpoint saved at iteration {i+1}: {checkpoint_file}")


    # Save final checkpoint
    final_checkpoint = os.path.join(checkpoint_dir, "final_checkpoint")
    algo.save(final_checkpoint)
    print(f"Final checkpoint saved: {final_checkpoint}")
    
    
    print("Training completed")
    algo.stop()
    ray.shutdown()
    if use_wandb:
        wandb.finish()
    print("Ray shutdown and Wandb finished")

def box_to_json(box):
    return {k: v for k, v in box.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', action='store', type=str, default='Overcooked-v1', help='Domain name')
    parser.add_argument('--n_agent', action='store', type=int, default=3, help='Number of agents')
    parser.add_argument('--grid_dim', action='store', type=int, nargs=2, default=[7,7], help='Grid world size')
    parser.add_argument('--task', action='store', type=int, default=6, help='The receipt agent cooks')
    parser.add_argument('--map_type', action='store', type=str, default="B", help='The type of map')
    parser.add_argument('--obs_radius', action='store', type=int, default=2, help='The radius of the agents')
    parser.add_argument('--mode', action='store', type=str, default="vector", help='The type of the observation(vector/image)')    
    parser.add_argument('--debug', action='store', type=bool, default=False, help='Whether print the debug information and render')
    parser.add_argument('--num_workers', action='store', type=int, default=2, help='Number of rollout workers')
    parser.add_argument('--num_gpus', action='store', type=int, default=3, help='Number of GPUs to use')
    parser.add_argument('--training_iterations', action='store', type=int, default=200, help='Number of training iterations')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='m3hf_overcooked', help='Wandb project name')
    parser.add_argument('--workers_per_cpu', type=int, default=2, help='Number of workers per CPU')

    args = parser.parse_args()
    
    env_config = {
        'env_id': args.env_id,  # Keep env_id in env_config for now
        'grid_dim': args.grid_dim,
        'task': TASKLIST[args.task],
        'rewardList': {"subtask finished": 10, "correct delivery": 200, "wrong delivery": -5, "step penalty": -0.1},
        'map_type': args.map_type,
        'n_agent': args.n_agent,
        'obs_radius': args.obs_radius,
        'mode': args.mode,
        'debug': args.debug,
    }

    train_multi_agent_ppo(env_config, args.num_workers, args.num_gpus, args.training_iterations, args.use_wandb, args.wandb_project)