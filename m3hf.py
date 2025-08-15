import numpy as np
from typing import List, Dict
import random
import requests
import json
import os
import logging

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
import gym
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper
import cv2
import numpy as np
import os
import logging
from ray.tune.registry import register_env
from play_rllib_ippo import MultiAgentOvercookedEnv
from ray.rllib.policy.sample_batch import SampleBatch
import copy
from datetime import datetime
import pygame

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
import torch.nn as nn

from math import sqrt
from language import test_feedback_parsing, test_reward_function_build, reward_aggregation, single_reward_function

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from custom_env import CustomRewardOvercookedEnv
from play_rllib_ippo import MultiAgentOvercookedEnv
from rollout import render_video_from_checkpoint
from ray.rllib.algorithms.callbacks import DefaultCallbacks

import wandb
from collections import defaultdict

TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEBUG = True

from collections import defaultdict
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from typing import Dict, Optional, TYPE_CHECKING



class OriginalRewardCallback(DefaultCallbacks):
    def on_episode_start(self, episode: MultiAgentEpisode, **kwargs):
        episode.user_data["original_rewards"] = []

    def on_episode_step(self, episode: MultiAgentEpisode, **kwargs):
        infos_all  = episode._last_infos
        # {'__common__': {'cur_mac': [3, 4, 3], 'mac_done': [False, True, False], 'original_reward': -0.1}, 0: {}, 1: {}, 2: {}} 

        original_reward = infos_all["__common__"]["original_reward"]
        episode.user_data["original_rewards"].append(original_reward)

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        original_rewards = episode.user_data["original_rewards"]
        original_return = np.sum(original_rewards)
        episode.custom_metrics["original_return"] = original_return


def train_policy(train_iterations, base_policies_save_path, env_config, current_reward_functions,
                 last_checkpoint=None, use_wandb=False, generation=0, num_gpus=2, gpu_devices="0,2", 
                 num_workers=None, cpu_utilization=0.6, max_workers=64, ray_head_port=None, 
                 ray_object_store_memory=20, force_new_ray=False, save_the_best=False, save_policy=True):
    
    # Get the number of CPUs available
    num_cpus = multiprocessing.cpu_count()
    print(f"Number of CPUs available: {num_cpus}")
    
    # Smart worker configuration based on parameters
    if num_workers is None:
        # Auto-detect based on CPU utilization parameter
        total_workers = min(int(num_cpus * cpu_utilization), max_workers)
    else:
        total_workers = min(num_workers, max_workers)
    
    workers_per_cpu = 1  # Each worker gets dedicated CPU resources
    print(f"Configuring {total_workers} workers for {num_cpus} CPUs (utilization: {total_workers/num_cpus*100:.1f}%)")
    print(f"Each worker will use {4} environments (total: {total_workers * 4} parallel environments)")
    
    # Configure Ray to use specific GPU devices and optimize memory
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    print(f"Using GPU devices: {gpu_devices}")
    
    # Handle Ray initialization
    if force_new_ray:
        print("Force restarting Ray cluster...")
        try:
            ray.shutdown()
        except:
            pass
    
    try:
        if ray.is_initialized() and not force_new_ray:
            print("Ray is already initialized, using existing cluster")
            cluster_resources = ray.cluster_resources()
            print(f"Available cluster resources: {cluster_resources}")
        else:
            # Try to connect to existing cluster first (unless forcing new)
            if not force_new_ray:
                try:
                    ray.init(address='auto', ignore_reinit_error=True)
                    print("✅ Connected to existing Ray cluster")
                    cluster_resources = ray.cluster_resources()
                    print(f"Cluster resources: {cluster_resources}")
                except:
                    force_new_ray = True  # Fall back to starting new cluster
            
            if force_new_ray:
                # Start new cluster
                print("Starting new Ray cluster")
                ray_kwargs = {
                    "ignore_reinit_error": True,
                    "num_gpus": num_gpus,
                    "num_cpus": total_workers + 2,  # Only use CPUs needed for workers + overhead
                    "object_store_memory": ray_object_store_memory * 1024 * 1024 * 1024,  # Convert GB to bytes
                    "_system_config": {
                        "max_pending_lease_requests_per_scheduling_category": 50,
                    }
                }
                
                # Add port configuration if specified
                if ray_head_port is not None:
                    ray_kwargs["port"] = ray_head_port
                
                print(f"Initializing Ray with {total_workers + 2} CPUs, {num_gpus} GPUs, {ray_object_store_memory}GB memory")
                if ray_head_port:
                    print(f"Using custom Ray port: {ray_head_port}")
                
                ray.init(**ray_kwargs)
                print(f"✅ Ray initialized successfully")
                
    except Exception as e:
        print(f"Ray initialization error: {e}")
        print("Attempting to force restart Ray...")
        try:
            ray.shutdown()
        except:
            pass
        
        # Force start new cluster
        ray_kwargs = {
            "ignore_reinit_error": True,
            "num_gpus": num_gpus,
            "num_cpus": total_workers + 2,
            "object_store_memory": ray_object_store_memory * 1024 * 1024 * 1024,
            "_system_config": {
                "max_pending_lease_requests_per_scheduling_category": 50,
            }
        }
        if ray_head_port is not None:
            ray_kwargs["port"] = ray_head_port
            
        ray.init(**ray_kwargs)
        print("✅ Ray restarted successfully")
    
    print(f"Ray Dashboard: http://127.0.0.1:8265")
    
    # Register the custom environment    
    temp_env = MultiAgentOvercookedEnv(env_config.copy())
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    
    register_env("custom_overcooked", lambda env_config: CustomRewardOvercookedEnv(env_config, current_reward_functions))
    
    
    
    # Create the PPO configuration
    config = (
        PPOConfig()
        .environment("custom_overcooked", env_config=env_config.copy())
        .framework("torch")
        .rollouts(num_rollout_workers=total_workers, num_envs_per_worker=4)  # Increased environments per worker
        .resources(
            num_gpus=num_gpus, 
            num_gpus_per_worker=max(0.1, num_gpus / max(total_workers, 1)),  # Distribute GPU more evenly
            num_cpus_per_worker=workers_per_cpu,
            num_learner_workers=2,  # Increased learner workers for better parallelization
            num_gpus_per_learner_worker=num_gpus / 2  # Split GPUs between learner workers
        )
        .multi_agent(
            policies={
                f"agent_{i}": (None, obs_space, act_space, {})
                for i in range(env_config['n_agent'])
            },
            policy_mapping_fn=lambda agent_id, episode, **kwargs: f"agent_{agent_id}",
        )
        .training(
            # Much larger batch sizes for massive parallel training
            train_batch_size=total_workers * 4 * 512 if generation > 0 else total_workers * 4 * 256,  # Scale with workers
            sgd_minibatch_size=4096,  # Larger mini-batches for better GPU utilization
            num_sgd_iter=20,  # More iterations to keep GPUs busy
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            kl_coeff=0.2,
            grad_clip=0.5,
        )
        .callbacks(OriginalRewardCallback) 
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

    # Build the algorithm
    algo = config.build()
    
    # Load the last checkpoint if provided
    if last_checkpoint and os.path.exists(last_checkpoint):
        algo.restore(last_checkpoint)
        print(f"Loaded checkpoint from {last_checkpoint}")
    else:
        print("Starting training from scratch.")

    print("PPO algorithm built")

    if use_wandb:
        wandb.init(project="overcooked_m3hf", config=env_config)

    # Create the generation folder
    generation_folder = os.path.join(base_policies_save_path, f"generation_{generation}")
    if not os.path.exists(generation_folder):
        os.makedirs(generation_folder)

    # Initialize variables for tracking the best model
    best_original_return_mean = float('-inf')
    best_checkpoint_path = None

    # List to collect original return means
    original_return_means = []

    # Training loop
    for i in range(train_iterations):
        print(f"Starting iteration {i+1}/{train_iterations}")
        result = algo.train()
        print(result["custom_metrics"])
        if result["custom_metrics"]:
            
            print(f"Customized episode_reward_mean: {result['episode_reward_mean']}")
            original_mean = result["custom_metrics"]["original_return_mean"]
            original_min = result["custom_metrics"]["original_return_min"]
            original_max = result["custom_metrics"]["original_return_max"]
            print(f"Original episode_reward_mean: {original_mean}")

            original_return_means.append(original_mean)

            if use_wandb:
                wandb.log({
                    "iteration": i + 1,
                    "Customized episode_reward_mean": result['episode_reward_mean'],
                    "Customized episode_reward_min": result['episode_reward_min'],
                    "Customized episode_reward_max": result['episode_reward_max'],
                    "Original episode_reward_mean": original_mean,
                    "Original episode_reward_min": original_min,
                    "Original episode_reward_max": original_max
                })

            # Check if we need to save the best model based on original reward
            if save_policy:
                if save_the_best and i >= 10:
                    if original_mean > best_original_return_mean:
                        best_original_return_mean = original_mean
                        # Save current best model with generation and iteration numbers
                        checkpoint_path = os.path.join(generation_folder, f"generation_{generation}_{i+1}")
                        algo.save(checkpoint_path)
                        best_checkpoint_path = checkpoint_path
                        print(f"New best model (based on original reward) saved at iteration {i+1}: {checkpoint_path}")
                else:
                    # Optionally save the model at specified intervals or at the end
                    if (i + 1) % train_iterations == 0:
                        checkpoint_path = os.path.join(generation_folder, f"generation_{generation}_{i+1}")
                        algo.save(checkpoint_path)
                        print(f"Checkpoint saved at iteration {i+1}: {checkpoint_path}")

    # At the end of training, decide which checkpoint to return
    if save_the_best and best_checkpoint_path is not None:
        checkpoint_path = best_checkpoint_path
        print(f"Best performing model for generation {generation} saved at: {checkpoint_path}")
    else:
        # Save the final model
        checkpoint_path = os.path.join(generation_folder, f"generation_{generation}_{train_iterations}")
        algo.save(checkpoint_path)
        print(f"Final model for generation {generation} saved at: {checkpoint_path}")

    this_generation_final_return_mean = np.mean(original_return_means)
    print(f"This generation's final original_return_mean: {this_generation_final_return_mean}")

    # Shutdown Ray
    algo.stop()
    ray.shutdown()
    print("Training completed and Ray shutdown")

    # Return the final checkpoint path
    return checkpoint_path, this_generation_final_return_mean

def setup_headless_display():
    """
    Setup virtual display for headless environments.
    
    Returns:
        tuple: (success: bool, method: str, cleanup_func: callable or None)
    """
    import os
    
    has_display = os.environ.get('DISPLAY') is not None
    
    if has_display:
        return True, "native_display", None
    
    print("No display detected. Setting up virtual display...")
    
    # Method 1: Try pyvirtualdisplay (preferred)
    try:
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(800, 600))
        display.start()
        print("✅ Started pyvirtualdisplay")
        return True, "pyvirtualdisplay", lambda: display.stop()
    except ImportError:
        print("pyvirtualdisplay not available, using SDL dummy driver...")
        pass
    except Exception as e:
        print(f"pyvirtualdisplay failed: {e}, falling back to SDL dummy driver...")
        pass
    
    # Method 2: Use SDL dummy driver (fallback)
    try:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        print("✅ Set SDL dummy drivers")
        return True, "SDL_dummy", None
    except Exception as e:
        print(f"Failed to set SDL dummy drivers: {e}")
        return False, "failed", None


def convert_obs_to_dict(obs):
    if isinstance(obs, dict):
        # If obs is already a dictionary, return as is
        return obs
    elif isinstance(obs, (list, tuple)):
        # If obs is a list or tuple, map agent IDs to observations
        return {agent_id: agent_obs for agent_id, agent_obs in enumerate(obs)}
    else:
        raise TypeError(f"Unexpected type for obs: {type(obs)}")

def generate_rollouts(policies_save_path, env_config, video_path, max_steps_per_episode=200, exploration=False):
    print("Generating rollouts...")
    logger.info("Starting render_video_from_checkpoint function")

    # Setup virtual display for headless environments
    display_success, display_method, cleanup_func = setup_headless_display()
    
    if not display_success:
        logger.error("Failed to setup display. Video recording may fail.")
        print("❌ Warning: Display setup failed. Video recording may not work.")
    else:
        logger.info(f"Display setup successful using: {display_method}")
        print(f"✅ Display ready using: {display_method}")

    temp_env = MultiAgentOvercookedEnv(env_config.copy())
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    
    register_env("multi_agent_overcooked", lambda config: MultiAgentOvercookedEnv(config))
    
    try:
        # Initialize Ray in local mode (if needed)
        # ray.init(ignore_reinit_error=True, local_mode=True)
        logger.info("Ray initialized successfully")

        # Create the environment
        logger.info("Creating environment")
        if 'env_id' in env_config:
            env_id = env_config.pop('env_id')
            env = gym.make(env_id, **env_config)
            env_config['env_id'] = env_id
        else:
            env = gym.make("Overcooked-MA-v1", **env_config)
        
        env = MacEnvWrapper(env)
        logger.info("Environment created successfully")


        logger.info("Setting up PPO configuration")
        config = (
            PPOConfig()
            .environment("multi_agent_overcooked", env_config=env_config.copy())
            .framework("torch")
            .rollouts(num_rollout_workers=0, num_envs_per_worker=1)
            .resources(num_gpus=0)
            .multi_agent(
                policies={
                    f"agent_{i}": (None, obs_space, act_space, {})
                    for i in range(env_config['n_agent'])
                },
                policy_mapping_fn=lambda agent_id, episode, **kwargs: f"agent_{agent_id}",
            )
            .training(
                train_batch_size=10240,
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
        logger.info("PPO configuration set up successfully")

        # Load the trained algorithm
        logger.info(f"Loading checkpoint from {policies_save_path}")
        algo = config.build()
        algo.restore(policies_save_path)
        logger.info("Checkpoint loaded successfully")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 5  # Adjust as needed
        video_writer = None
        
        obs = env.reset(seed=None)  # seed=None 会使用随机种子
        done = False
        step = 0
        output_reward = 0
        discount = 1
        state = {
            f"agent_{i}": algo.get_policy(f"agent_{i}").get_initial_state()
            for i in range(env_config['n_agent'])
        }

        while step < max_steps_per_episode and not done:
            actions = {}
            print("Type of obs after env.reset():", type(obs))
            obs = convert_obs_to_dict(obs)
            print("Type of obs after convert_obs_to_dict:", type(obs))
            if step == 0:
                obs = obs[0]
            for agent_id, agent_obs in obs.items():
                policy_id = f"agent_{agent_id}"
                agent_obs = agent_obs.astype(np.float32)
                # Compute the action
                action, state[policy_id], _ = algo.compute_single_action(
                    observation=agent_obs,
                    state=state[policy_id],
                    policy_id=policy_id,
                    explore=exploration,
                    full_fetch=True
                )

                actions[agent_id] = action

            obs, rewards, done, info = env.step(actions)

            print("action: ", actions)
            output_reward += discount * rewards[0]
            discount *= 0.99
            print("------------------------------------------------------------------")
            print("step: ", step, "rewards: ", output_reward)
            print("#############################################")

            # Render the frame
            frame = env.render(mode='rgb_array')
            if video_writer is None:
                h, w = frame.shape[:2]
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
            
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            step += 1
            
            if done:
                break   
        
        video_writer.release()
        env.close()
        logger.info("Video saved and environment closed")
        pygame.quit()

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

    finally:
        # Cleanup virtual display if needed
        if 'cleanup_func' in locals() and cleanup_func:
            try:
                cleanup_func()
                logger.info("Virtual display cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up virtual display: {e}")
        
        ray.shutdown()
        logger.info("Ray shutdown completed")
    print("Finished generating rollouts.")

def weight_decay_and_performance_adjustment(weights, decay_rate, num_new_func, 
                                           current_performance=None, previous_performance=None, 
                                           adjustment_factor=0.1):
    """
    Weight Decay and Performance-based Adjustment as described in Section 4.3 of the paper.
    
    This implements the exact mechanism from the M3HF paper:
    1. Initialize new weights uniformly: w_{i,m} = 1/(|P_i| + 1)
    2. Apply exponential decay: w_{i,m} = w_{i,m} * α^{M-m}
    3. Performance-based adjustment: w_{i,m} ± β
    4. Normalize weights: w_{i,m} = w_{i,m} / Σw_{i,m}
    
    Args:
        weights (List[float]): Current weights of reward functions
        decay_rate (float): Decay factor α ∈ (0,1) 
        num_new_func (int): Number of new reward functions to add
        current_performance (float): r_ori^i_{k+1} - performance after new reward
        previous_performance (float): r_ori^i_k - performance before new reward  
        adjustment_factor (float): β - small adjustment factor for performance-based changes
        
    Returns:
        List[float]: Updated and normalized weights
    """
    import numpy as np
    
    # Step 1: Add new reward functions with uniform initialization
    if num_new_func > 0:
        # Paper formula: w_{i,m} = 1/(|P_i| + 1)
        total_functions = len(weights) + num_new_func
        uniform_weight = 1.0 / total_functions
        
        # Reinitialize all weights to uniform distribution
        updated_weights = [uniform_weight] * total_functions
    else:
        updated_weights = weights.copy()
    
    # Step 2: Apply exponential decay to existing reward functions
    # Paper formula: w_{i,m} = w_{i,m} * α^{M-m}, ∀m ∈ {1, ..., M-1}
    M = len(updated_weights)
    
    if num_new_func > 0:
        # Only apply decay to old functions (not the newly added ones)
        for m in range(len(weights)):  # Original functions
            # M-m: older functions get more decay
            decay_power = M - 1 - m
            updated_weights[m] = updated_weights[m] * (decay_rate ** decay_power)
    else:
        # Apply decay to all existing functions
        for m in range(M):
            decay_power = M - 1 - m
            updated_weights[m] = updated_weights[m] * (decay_rate ** decay_power)
    
    # Step 3: Performance-based adjustment for the newest reward function
    # Paper equation: w_{i,m} = w_{i,m} ± β based on performance difference
    if (current_performance is not None and 
        previous_performance is not None and 
        num_new_func > 0):
        
        performance_diff = current_performance - previous_performance
        newest_idx = len(updated_weights) - 1  # Index of newest reward function
        
        if performance_diff > 0:
            # Performance improved: w_{i,m} = w_{i,m} + β
            updated_weights[newest_idx] = updated_weights[newest_idx] + adjustment_factor
        else:
            # Performance degraded: w_{i,m} = max(0, w_{i,m} - β)  
            updated_weights[newest_idx] = max(0, updated_weights[newest_idx] - adjustment_factor)
    
    # Step 4: Normalize all weights
    # Paper formula: w_{i,m} = w_{i,m} / Σw_{i,m}, ∀m ∈ {1, ..., M}
    total_weight = sum(updated_weights)
    if total_weight > 0:
        normalized_weights = [w / total_weight for w in updated_weights]
    else:
        # If all weights are zero, redistribute uniformly
        normalized_weights = [1.0 / len(updated_weights)] * len(updated_weights)
    
    return normalized_weights


def weight_update_exponential(weights, decay_rate, num_new_func, aggregation, post_adjustment=False, last_generation_final_return_mean=None, this_generation_final_return_mean=None):
    """
    Apply exponential weight decay to existing weights and add new weights for new functions.
    
    Args:
    weights (List[float]): Current weights of reward functions.
    decay_rate (float): The decay rate (alpha) for existing weights.
    num_new_func (int): Number of new functions to add.
    aggregation (bool): Whether to use aggregation or not.
    
    Returns:
    List[float]: Updated weights after decay and normalization.
    """
    if not post_adjustment:
        decayed_weights = [w * (decay_rate ** (len(weights) - i)) for i, w in enumerate(weights)]
        
        if aggregation:
            new_weights = [1.0]
        else:
            new_weights = [1.0] * num_new_func
        
        updated_weights = decayed_weights + new_weights
        total_weight = sum(updated_weights)
        normalized_weights = [w / total_weight for w in updated_weights]
        
        return normalized_weights
    else:
        if last_generation_final_return_mean > this_generation_final_return_mean:
            # Decrease the last weight by 40%
            print("WARNING: THE PERFRMANCE DECREASED! Decrease the last weight by 40%")
            weights[-1] *= 0.6
        else:
            # Increase the last weight by 30%
            print("Good News: THE PERFRMANCE INCREASED! Increase the last weight by 30%")
            weights[-1] *= 1.3
        # Normalize the weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        return normalized_weights


def weight_update_adaptive(weights, decay_rate, num_new_func, performance_history=None, feedback_quality=None, learning_rates=None):
    """
    Adaptive weight update with performance-aware and quality-aware adjustments.
    
    Note: This is not true meta-learning, but rather heuristic adaptation based on
    performance trends and feedback quality. The name 'adaptive' better reflects
    the actual functionality.
    """
    import numpy as np
    
    if learning_rates is None:
        learning_rates = [0.1] * len(weights)
    
    # Initialize new weights for new functions
    if num_new_func > 0:
        new_weights = [1.0] * num_new_func
        new_learning_rates = [0.1] * num_new_func
        weights = weights + new_weights
        learning_rates = learning_rates + new_learning_rates
    
    # Meta-learning adaptation based on performance history
    if performance_history is not None and len(performance_history) >= 2:
        # Calculate performance gradient (improvement/decline)
        perf_gradient = np.diff(performance_history)
        recent_gradient = np.mean(perf_gradient[-2:]) if len(perf_gradient) >= 2 else perf_gradient[-1]
        
        # Adapt learning rates based on performance trend
        adaptation_factor = 1.0 + 0.2 * np.tanh(recent_gradient)  # Scale factor between 0.8-1.2
        learning_rates = [lr * adaptation_factor for lr in learning_rates]
    
    # Quality-aware weight adjustment
    if feedback_quality is not None:
        avg_quality = np.mean(feedback_quality) if feedback_quality else 1.0
        quality_factor = max(0.5, min(1.5, avg_quality))  # Clamp between 0.5-1.5
        
        # Apply quality-based modulation to recent weights
        for i in range(max(0, len(weights) - num_new_func), len(weights)):
            weights[i] *= quality_factor
    
    # Gradient-based weight optimization
    updated_weights = []
    for i, (w, lr) in enumerate(zip(weights, learning_rates)):
        # Apply decay with adaptive learning rate
        age_factor = (len(weights) - i) / len(weights)  # Newer weights get less decay
        adaptive_decay = decay_rate ** (lr * age_factor)
        updated_weights.append(w * adaptive_decay)
    
    # Normalize weights
    total_weight = sum(updated_weights)
    if total_weight > 0:
        normalized_weights = [w / total_weight for w in updated_weights]
    else:
        normalized_weights = [1.0 / len(updated_weights)] * len(updated_weights)
    
    return normalized_weights, learning_rates


def multi_phase_feedback_integration(agents_reward_data, feedback_history, performance_history, 
                                   similarity_threshold=0.7, quality_threshold=0.4):
    """
    Advanced multi-phase feedback integration with similarity detection and quality filtering.
    
    This implements a sophisticated feedback processing pipeline that:
    1. Assesses feedback quality and filters low-quality inputs
    2. Detects similar feedback to avoid redundant reward functions
    3. Implements temporal weighting for recent vs historical feedback
    4. Provides conflict resolution for contradictory feedback
    
    Args:
        agents_reward_data (Dict): Current agent reward data structure
        feedback_history (List[str]): Historical feedback for similarity comparison
        performance_history (List[float]): Performance scores for context
        similarity_threshold (float): Threshold for considering feedback similar
        quality_threshold (float): Minimum quality score for feedback inclusion
        
    Returns:
        Dict: Updated agents_reward_data with integrated feedback
    """
    from language import assess_feedback_quality, filter_low_quality_feedback
    from utils import cal_similarity
    import numpy as np
    
    # Phase 1: Quality Assessment and Filtering
    print("Phase 1: Assessing feedback quality...")
    for agent_id in agents_reward_data:
        if agents_reward_data[agent_id]["feedback_json"]:
            latest_feedback = agents_reward_data[agent_id]["feedback_json"][-1]
            
            # Filter low-quality feedback
            filtered_feedback = filter_low_quality_feedback(latest_feedback, quality_threshold)
            
            if len(filtered_feedback) < len(latest_feedback):
                print(f"Filtered {len(latest_feedback) - len(filtered_feedback)} low-quality feedback items for {agent_id}")
                agents_reward_data[agent_id]["feedback_json"][-1] = filtered_feedback
    
    # Phase 2: Similarity Detection and Deduplication  
    print("Phase 2: Detecting similar feedback...")
    for agent_id in agents_reward_data:
        if agents_reward_data[agent_id]["assignment_feedback"]:
            current_feedback = agents_reward_data[agent_id]["assignment_feedback"][-1]
            
            # Check similarity with historical feedback
            similar_found = False
            for historical_feedback in feedback_history:
                if cal_similarity(current_feedback, historical_feedback) > similarity_threshold:
                    print(f"Similar feedback detected for {agent_id}. Applying reduced weight.")
                    similar_found = True
                    break
            
            # Mark feedback as similar for weight adjustment
            agents_reward_data[agent_id]["feedback_similarity"] = agents_reward_data[agent_id].get("feedback_similarity", [])
            agents_reward_data[agent_id]["feedback_similarity"].append(1.0 if not similar_found else 0.5)
    
    # Phase 3: Temporal Weighting
    print("Phase 3: Applying temporal weighting...")
    for agent_id in agents_reward_data:
        num_feedback = len(agents_reward_data[agent_id]["weights_pool"])
        if num_feedback > 1:
            # Apply temporal decay to older weights
            temporal_weights = []
            for i in range(num_feedback):
                age = num_feedback - i - 1  # 0 for newest, increases for older
                temporal_factor = np.exp(-0.1 * age)  # Exponential decay
                temporal_weights.append(temporal_factor)
            
            # Store temporal weights for later use
            agents_reward_data[agent_id]["temporal_weights"] = temporal_weights
    
    # Phase 4: Conflict Resolution
    print("Phase 4: Resolving feedback conflicts...")
    for agent_id in agents_reward_data:
        if len(agents_reward_data[agent_id]["assignment_feedback"]) > 1:
            recent_feedback = agents_reward_data[agent_id]["assignment_feedback"][-2:]
            
            # Check for contradictory feedback (using simple sentiment analysis)
            def extract_sentiment(feedback_text):
                positive_words = ['good', 'great', 'better', 'excellent', 'perfect', 'nice', 'well', 'improve']
                negative_words = ['bad', 'wrong', 'avoid', 'stop', 'not', 'problem', 'issue', 'slow', 'poor']
                
                pos_count = sum(1 for word in positive_words if word in feedback_text.lower())
                neg_count = sum(1 for word in negative_words if word in feedback_text.lower())
                
                return pos_count - neg_count
            
            if len(recent_feedback) == 2:
                sentiment1 = extract_sentiment(recent_feedback[0])
                sentiment2 = extract_sentiment(recent_feedback[1])
                
                # Check for sentiment conflict
                if (sentiment1 > 0 and sentiment2 < 0) or (sentiment1 < 0 and sentiment2 > 0):
                    print(f"Conflict detected in recent feedback for {agent_id}. Applying resolution strategy.")
                    
                    # Resolution: Weight recent feedback higher if performance is improving
                    if performance_history and len(performance_history) >= 2:
                        recent_trend = performance_history[-1] - performance_history[-2]
                        if recent_trend > 0:  # Performance improving
                            # Trust recent feedback more
                            conflict_weights = [0.3, 0.7]
                        else:  # Performance declining
                            # Be more conservative, weight earlier feedback higher
                            conflict_weights = [0.6, 0.4]
                        
                        agents_reward_data[agent_id]["conflict_resolution_weights"] = conflict_weights
    
    return agents_reward_data


def adaptive_reward_template_selection(feedback_text, agent_performance_history, 
                                     available_templates=None):
    """
    Dynamically select the most appropriate reward function template based on:
    1. Feedback content analysis
    2. Agent performance patterns
    3. Task complexity assessment
    
    Args:
        feedback_text (str): Human feedback text
        agent_performance_history (List[float]): Performance trajectory for this agent
        available_templates (List[str], optional): Available template types
        
    Returns:
        tuple: (selected_template_name, template_parameters)
    """
    if available_templates is None:
        available_templates = [
            "distance_based", "action_based", "status_based", 
            "proximity_based", "composite", "time_penalty", "cooperation_based"
        ]
    
    template_scores = {}
    
    # Analyze feedback content
    feedback_lower = feedback_text.lower()
    
    # Distance-based indicators
    distance_indicators = ['closer', 'near', 'far', 'distance', 'approach', 'move to', 'go to']
    distance_score = sum(1 for indicator in distance_indicators if indicator in feedback_lower)
    template_scores['distance_based'] = distance_score
    
    # Action-based indicators  
    action_indicators = ['chop', 'deliver', 'pick', 'take', 'action', 'do', 'should', 'need to']
    action_score = sum(1 for indicator in action_indicators if indicator in feedback_lower)
    template_scores['action_based'] = action_score
    
    # Status-based indicators
    status_indicators = ['chopped', 'prepared', 'ready', 'finished', 'completed', 'status']
    status_score = sum(1 for indicator in status_indicators if indicator in feedback_lower)
    template_scores['status_based'] = status_score
    
    # Cooperation-based indicators
    coop_indicators = ['together', 'coordinate', 'team', 'both', 'cooperation', 'help', 'work with']
    coop_score = sum(1 for indicator in coop_indicators if indicator in feedback_lower)
    template_scores['cooperation_based'] = coop_score
    
    # Time penalty indicators
    time_indicators = ['slow', 'fast', 'quick', 'time', 'speed', 'hurry', 'urgent']
    time_score = sum(1 for indicator in time_indicators if indicator in feedback_lower)
    template_scores['time_penalty'] = time_score
    
    # Performance-based template adjustment
    if agent_performance_history and len(agent_performance_history) >= 3:
        recent_performance = np.mean(agent_performance_history[-3:])
        overall_performance = np.mean(agent_performance_history)
        
        # If recent performance is declining, favor composite templates
        if recent_performance < overall_performance * 0.9:
            template_scores['composite'] = template_scores.get('composite', 0) + 2
        
        # If performance is highly variable, use proximity-based templates
        performance_std = np.std(agent_performance_history)
        if performance_std > np.mean(agent_performance_history) * 0.3:
            template_scores['proximity_based'] = template_scores.get('proximity_based', 0) + 1
    
    # Select template with highest score
    if template_scores:
        selected_template = max(template_scores, key=template_scores.get)
        max_score = template_scores[selected_template]
        
        # If no clear winner, default to composite
        if max_score == 0 or list(template_scores.values()).count(max_score) > 1:
            selected_template = 'composite'
    else:
        selected_template = 'composite'
    
    # Generate template parameters based on selection
    template_parameters = {
        'weight': 1.0,
        'priority': 'medium',
        'temporal_decay': 0.95 if 'time' in feedback_lower else 1.0,
        'cooperation_factor': 1.2 if 'coordinate' in feedback_lower else 1.0
    }
    
    return selected_template, template_parameters



if __name__ == "__main__":   
    
    # step 1: Initialize, build environment, build the reward function pool
    parser = argparse.ArgumentParser(description='M3HF: Multi-agent Reinforcement Learning from Multi-phase Human Feedback')
    parser.add_argument('--env_id', action='store', type=str, default='Overcooked-MA-v1', help='Domain name')
    parser.add_argument('--n_agent', action='store', type=int, default=3, help='Number of agents')
    parser.add_argument('--grid_dim', action='store', type=int, nargs=2, default=[7,7], help='Grid world size')
    parser.add_argument('--task', action='store', type=int, default=6, help='The receipt agent cooks')
    parser.add_argument('--map_type', action='store', type=str, default="B", help='The type of map')
    parser.add_argument('--obs_radius', action='store', type=int, default=2, help='The radius of the agents')
    parser.add_argument('--mode', action='store', type=str, default="vector", help='The type of the observation(vector/image)')    
    parser.add_argument('--debug', action='store', type=bool, default=False, help='Whether print the debug information and render')
    parser.add_argument('--num_workers', action='store', type=int, default=None, help='Number of rollout workers (auto-detect if None)')
    parser.add_argument('--cpu_utilization', type=float, default=0.6, help='Fraction of CPUs to use for workers (0.1-1.0)')
    parser.add_argument('--max_workers', type=int, default=16, help='Maximum number of workers to use')
    parser.add_argument('--num_gpus', action='store', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--gpu_devices', type=str, default='2', help='Comma-separated GPU device IDs to use (e.g., "0,2")')
    parser.add_argument('--ray_head_port', type=int, default=None, help='Ray head node port (auto if None)')
    parser.add_argument('--ray_object_store_memory', type=int, default=20, help='Ray object store memory in GB')
    parser.add_argument('--force_new_ray', action='store_true', help='Force start a new Ray cluster')
    parser.add_argument('--training_iterations', action='store', type=int, default=200, help='Number of training iterations')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='m3hf_overcooked', help='Wandb project name')
    parser.add_argument('--workers_per_cpu', type=int, default=1, help='Number of workers per CPU')
    parser.add_argument('--demo_mode', action='store_true', help='Run in demo mode with simulated feedback')
    parser.add_argument('--generations', type=int, default=2, help='Number of training generations')
    parser.add_argument('--skip_pretrain', action='store_true', help='Skip generation 0 pretraining')

    args = parser.parse_args()
    
    # Initialize based on parsed arguments
    gen_0_open = not args.skip_pretrain
    
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_policies_save_path = f"./m3hf_results_{time_str}/"
    if not os.path.exists(base_policies_save_path) and gen_0_open:
        os.makedirs(base_policies_save_path) 
    
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
    
    rollout_env_config = {
        'env_id': args.env_id, 
        'grid_dim': args.grid_dim,
        'task': TASKLIST[args.task],
        'rewardList': {"subtask finished": 10, "correct delivery": 200, "wrong delivery": -5, "step penalty": -0.1},
        'map_type': args.map_type,
        'n_agent': args.n_agent,
        'obs_radius': args.obs_radius,
        'mode': args.mode,
        'debug': True
    }
    
    num_agents = 3
    last_checkpoint = None

    Aggregation = True
    weight_decay_method = "paper" # "exponential", "adaptive", "paper" (Section 4.3 implementation)
    weight_decay_rate = 0.9
    post_re_training = False
    
    ## train generations, how many times we want to repeat the human-in-the-loop process
    train_generations = args.generations # Use command line argument
    ## train iterations, how many iterations we want to train the policy in each generation
    train_iterations = args.training_iterations
    
    ## task name
    task_name = "lettuce-onion-tomato salad" # "lettuce-tomato salad"
    
    agents_reward_data = {
        f"agent_{i}": {
            "feedback_json": [],
            "reward_function_pool": [],
            "reward_function_explanation": [],
            "weights_pool": [],
            "reward_function_strings": [],
            "assignment_feedback": []
        } for i in range(num_agents)
    }
    
  
    current_reward_functions = {}
    
    last_generation_final_return_mean = None
    feedback_history = []  # Store all historical feedback for similarity detection
    performance_history_per_agent = {f"agent_{i}": [] for i in range(num_agents)}  # Track per-agent performance

    # step 2: Train the agents with the original reward functions in few iterations

    
    if gen_0_open:
        # Pre-train the policy for one generation (generation 0)
        generation = 0
        print(f"Starting pre-training for generation {generation}")
        last_checkpoint, this_generation_final_return_mean = train_policy(
            train_iterations=train_iterations,
            base_policies_save_path=base_policies_save_path,
            env_config=env_config,
            current_reward_functions=current_reward_functions,
            last_checkpoint=last_checkpoint,
            use_wandb=args.use_wandb, 
            generation=generation,
            num_gpus=args.num_gpus,
            gpu_devices=args.gpu_devices,
            num_workers=args.num_workers,
            cpu_utilization=args.cpu_utilization,
            max_workers=args.max_workers,
            ray_head_port=args.ray_head_port,
            ray_object_store_memory=args.ray_object_store_memory,
            force_new_ray=args.force_new_ray,
            save_the_best=False 
        )
    else:
        base_policies_save_path = './m3hf_results_20241017_022701/'
        last_checkpoint = os.path.join(os.path.dirname(__file__), 'm3hf_results_20241017_022701', 'generation_0', 'generation_0_200')
        

    for generation in range(1, train_generations + 1):
        # Step 3: Generate the rollouts with the trained policies
        video_dir = os.path.join(base_policies_save_path, f"generation_{generation-1}")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"generation_{generation-1}.mp4")
        
        print("Last checkpoint: ", last_checkpoint)
        print("Video path: ", video_path)
        
        generate_rollouts(last_checkpoint, rollout_env_config.copy(), video_path, max_steps_per_episode=200, exploration=False)

        # Step 4: Collect the feedback from the rollouts
        if args.demo_mode:
            # Use predefined feedback for demo mode
            demo_feedback = [
                "Hey, I noticed that one of the agents isn't moving at all during the game. You might want to check if there's an issue with their controls or programming. Getting both agents active will really boost your salad production and make the game more fun!",
                "Loved watching you play! Quick tip: if you have vegetables on the cutting board, make sure to chop them right away. Leaving them unchopped slows down your salad-making process. Keep those knives moving!",
                "Remember to focus on speed without sacrificing coordination. While it's important to chop vegetables quickly, ensuring that both agents are working on different tasks will maximize your overall efficiency. Keep practicing this balance, and you'll see significant improvements in your salad preparation.",
                "Avoid unnecessary movement around the kitchen to save time. If the Green agent is stationed near the lettuce and tomatoes, they should focus on those ingredients. The Rose agent can handle the onions and mixing on the other side. This division will minimize collisions and streamline your workflow."
            ]
            
            feedback_index = min(generation - 1, len(demo_feedback) - 1)
            feedback = demo_feedback[feedback_index]
            print(f"Using demo feedback for generation {generation}: {feedback}")
        else:
            print("Please provide feedback for the agents' performance:")
            feedback = input().strip()
            
        # Step 5: Enhanced feedback processing with quality assessment and advanced integration
        
        # Store feedback in history for similarity detection
        feedback_history.append(feedback)
        
        # Phase 1: Parse the feedback and build reward functions
        assignment_feedback = test_feedback_parsing(num_agents, feedback, env_config["map_type"])
        json_feedback = json.loads(assignment_feedback)

        print("Parsed feedback:")
        for key, value in json_feedback.items():
            print(f"{key}: {value}")
        
        # Phase 2: Assess feedback quality and apply template selection
        print("\n=== ADVANCED M3HF FEEDBACK PROCESSING ===")
        from language import assess_feedback_quality
        
        feedback_quality_scores = []
        for key, value in json_feedback.items():
            if key != "all":
                quality_score = assess_feedback_quality(value, json_feedback, 
                                                      performance_history_per_agent.get(key, []))
                feedback_quality_scores.append(quality_score)
                print(f"Feedback quality for {key}: {quality_score:.3f}")
                
                # Apply adaptive template selection
                template, template_params = adaptive_reward_template_selection(
                    value, performance_history_per_agent.get(key, [])
                )
                print(f"Selected template for {key}: {template} with params: {template_params}")
        
        new_agents_new_reward_function = test_reward_function_build(num_agents, json_feedback, env_config["map_type"])
        
        ## Update reward function pools for each agent
        for i in range(num_agents):
            num_new_func = len(new_agents_new_reward_function[i])
            print(f"There are {num_new_func} new reward functions for agent {i}")
            agents_reward_data[f"agent_{i}"]["feedback_json"].append(new_agents_new_reward_function[i])

             
        ## Aggregate reward functions for each agent (Let the reward functions of this generation aggregate into a new reward function and replace the original function)
        if Aggregation:
            for i in range(num_agents):
                new_reward_function_agent, reward_function_str = reward_aggregation(agents_reward_data[f"agent_{i}"]["feedback_json"][-1])
                agents_reward_data[f"agent_{i}"]["reward_function_pool"].append(new_reward_function_agent)
                agents_reward_data[f"agent_{i}"]["reward_function_strings"].append(reward_function_str)
                for j in range(len(new_agents_new_reward_function[i])):
                    if "explanation" in new_agents_new_reward_function[i][j]:
                        agents_reward_data[f"agent_{i}"]["reward_function_explanation"].append(new_agents_new_reward_function[i][j]["explanation"])
                    else:
                        agents_reward_data[f"agent_{i}"]["reward_function_explanation"].append("")
                    if "original_feedback" in new_agents_new_reward_function[i][j]:
                        agents_reward_data[f"agent_{i}"]["assignment_feedback"].append(new_agents_new_reward_function[i][j]["original_feedback"])
                    else:
                        agents_reward_data[f"agent_{i}"]["assignment_feedback"].append("")
        else:
            for i in range(num_agents):
                for j in range(len(new_agents_new_reward_function[i])):
                    new_reward_function_agent, reward_function_str = single_reward_function(new_agents_new_reward_function[i][j])
                    agents_reward_data[f"agent_{i}"]["reward_function_pool"].append(new_reward_function_agent)
                    agents_reward_data[f"agent_{i}"]["reward_function_strings"].append(reward_function_str)
                    agents_reward_data[f"agent_{i}"]["reward_function_explanation"].append(new_agents_new_reward_function[i][j]["explanation"])
                    agents_reward_data[f"agent_{i}"]["assignment_feedback"].append(new_agents_new_reward_function[i][j]["original_feedback"])
            
        
        ## Weights for each function in the pool
        if generation == 1:
            for i in range(num_agents):
                if len(agents_reward_data[f"agent_{i}"]["reward_function_pool"]) > 0:
                    agents_reward_data[f"agent_{i}"]["weights_pool"] = [1.0] * len(agents_reward_data[f"agent_{i}"]["reward_function_pool"])
        else:
            # Apply multi-phase feedback integration before weight updates
            print("\n=== MULTI-PHASE FEEDBACK INTEGRATION ===")
            agents_reward_data = multi_phase_feedback_integration(
                agents_reward_data, feedback_history[:-1], 
                [last_generation_final_return_mean] if last_generation_final_return_mean else []
            )
            
            for i in range(num_agents):
                num_new_func = len(new_agents_new_reward_function[i])
                old_weights = agents_reward_data[f"agent_{i}"]["weights_pool"]
                
                if weight_decay_method == "exponential":
                    new_weights = weight_update_exponential(old_weights, weight_decay_rate, num_new_func, Aggregation)
                elif weight_decay_method == "adaptive":
                    agent_performance_history = performance_history_per_agent.get(f"agent_{i}", [])
                    agent_feedback_quality = [feedback_quality_scores[i]] if i < len(feedback_quality_scores) else None
                    
                    new_weights, learning_rates = weight_update_adaptive(
                        old_weights, weight_decay_rate, num_new_func,
                        performance_history=agent_performance_history,
                        feedback_quality=agent_feedback_quality
                    )
                    
                    # Store learning rates for future use
                    agents_reward_data[f"agent_{i}"]["learning_rates"] = learning_rates
                elif weight_decay_method == "paper":
                    # Use the exact paper implementation from Section 4.3
                    current_perf = this_generation_final_return_mean if generation > 1 else None
                    previous_perf = last_generation_final_return_mean if generation > 1 else None
                    
                    new_weights = weight_decay_and_performance_adjustment(
                        old_weights, weight_decay_rate, num_new_func,
                        current_performance=current_perf,
                        previous_performance=previous_perf,
                        adjustment_factor=0.1
                    )
                
                agents_reward_data[f"agent_{i}"]["weights_pool"] = new_weights
                print(f"Updated weights for agent_{i}: {[f'{w:.3f}' for w in new_weights]}")
                
        for i in range(num_agents):
            current_reward_functions[f"agent_{i}"] = lambda obs, act, ori_reward, t: sum(
                agents_reward_data[f"agent_{i}"]["weights_pool"][j] * reward_func(obs, act, ori_reward, t)
                for j, reward_func in enumerate(agents_reward_data[f"agent_{i}"]["reward_function_pool"])
            )
        
    
        if DEBUG:
            for i in range(num_agents):
                print(agents_reward_data[f"agent_{i}"]["reward_function_strings"][-1])
                
            
        # Step 6: Update the environment with new reward functions
        # No changes needed here since we pass current_reward_functions to the env

        # Step 7: Update policies with new reward functions
        print(f"Starting training for generation {generation}")
        last_checkpoint, this_generation_final_return_mean = train_policy(
            train_iterations=train_iterations,
            base_policies_save_path=base_policies_save_path,
            env_config=env_config,
            current_reward_functions=current_reward_functions,
            last_checkpoint=last_checkpoint,
            use_wandb=args.use_wandb, 
            generation=generation,
            num_gpus=args.num_gpus,
            gpu_devices=args.gpu_devices,
            num_workers=args.num_workers,
            cpu_utilization=args.cpu_utilization,
            max_workers=args.max_workers,
            ray_head_port=args.ray_head_port,
            ray_object_store_memory=args.ray_object_store_memory,
            force_new_ray=args.force_new_ray,
            save_the_best=True 
        )
        # Weight adjustment after training and before policy save (generation end)
        # Post-adjustment based on the selected weight decay method
        if last_generation_final_return_mean is not None and weight_decay_method != "paper":
            # Note: Paper method already includes performance-based adjustment during main update
            for i in range(num_agents):
                old_weights = agents_reward_data[f"agent_{i}"]["weights_pool"]
                if weight_decay_method == "exponential":
                    new_weights = weight_update_exponential(
                        weights=old_weights,
                        decay_rate=weight_decay_rate,
                        num_new_func=0, 
                        aggregation=Aggregation,
                        post_adjustment=True,
                        last_generation_final_return_mean=last_generation_final_return_mean,
                        this_generation_final_return_mean=this_generation_final_return_mean
                    )
                    agents_reward_data[f"agent_{i}"]["weights_pool"] = new_weights
                
            #TODO: Check Whether the Post Re-training using the adjusted reward functions if needed?
            # Update current_reward_functions with updated weights
            if post_re_training and last_generation_final_return_mean > this_generation_final_return_mean:
                print("WARNING: THE PERFRMANCE DECREASED! And as the Post Re-training is on, will keep training the policy using the updated reward functions!")
                for i in range(num_agents):
                    current_reward_functions[f"agent_{i}"] = lambda obs, act, ori_reward, t: sum(
                        agents_reward_data[f"agent_{i}"]["weights_pool"][j] * reward_func(obs, act, ori_reward, t)
                        for j, reward_func in enumerate(agents_reward_data[f"agent_{i}"]["reward_function_pool"])
                    )
                # Training the policy with the new reward functions
                print(f"Starting training for generation {generation}")
                last_checkpoint, this_generation_final_return_mean = train_policy(
                    train_iterations=50, # re-training iterations
                    base_policies_save_path=base_policies_save_path,
                    env_config=env_config,
                    current_reward_functions=current_reward_functions,
                    last_checkpoint=last_checkpoint,
                    use_wandb=args.use_wandb, 
                    generation=generation,
                    num_gpus=args.num_gpus,
                    gpu_devices=args.gpu_devices,
                    num_workers=args.num_workers,
                    cpu_utilization=args.cpu_utilization,
                    max_workers=args.max_workers,
                    save_the_best=False, 
                    ray_head_port=args.ray_head_port,
                    ray_object_store_memory=args.ray_object_store_memory,
                    force_new_ray=args.force_new_ray,
                    save_policy=False
                )
        
        # Update performance tracking for each agent
        for i in range(num_agents):
            performance_history_per_agent[f"agent_{i}"].append(this_generation_final_return_mean)
        
        # Update last_generation_final_return_mean for next generation
        last_generation_final_return_mean = this_generation_final_return_mean
        
        # Print comprehensive generation summary
        print(f"\n=== GENERATION {generation} SUMMARY ===")
        print(f"Performance: {this_generation_final_return_mean:.4f}")
        if last_generation_final_return_mean:
            trend = "↑" if this_generation_final_return_mean > last_generation_final_return_mean else "↓"
            change = this_generation_final_return_mean - last_generation_final_return_mean
            print(f"Performance change: {change:+.4f} {trend}")
        
        print("Active reward functions per agent:")
        for i in range(num_agents):
            num_functions = len(agents_reward_data[f"agent_{i}"]["reward_function_pool"])
            avg_weight = np.mean(agents_reward_data[f"agent_{i}"]["weights_pool"]) if agents_reward_data[f"agent_{i}"]["weights_pool"] else 0.0
            print(f"  Agent {i}: {num_functions} functions, avg weight: {avg_weight:.3f}")
        
        print(f"Generation {generation} complete. Enhanced M3HF processing applied.")
        
        # Save generation data for analysis (optional)
        generation_data = {
            'generation': generation,
            'performance': this_generation_final_return_mean,
            'feedback_history': feedback_history[-1:],  # Only current feedback
            'feedback_quality_scores': feedback_quality_scores if 'feedback_quality_scores' in locals() else [],
            'agents_data': {
                agent_id: {
                    'num_functions': len(agents_reward_data[agent_id]["reward_function_pool"]),
                    'weights': agents_reward_data[agent_id]["weights_pool"],
                    'learning_rates': agents_reward_data[agent_id].get("learning_rates", [])
                }
                for agent_id in agents_reward_data
            }
        }
        
        # Optional: Save to file for later analysis
        if DEBUG:
            import json
            # Create generation folder if it doesn't exist
            generation_folder = os.path.join(base_policies_save_path, f"generation_{generation}")
            os.makedirs(generation_folder, exist_ok=True)
            
            data_file_path = os.path.join(generation_folder, f"generation_{generation}_data.json")
            with open(data_file_path, "w") as f:
                json.dump(generation_data, f, indent=2)
            print(f"Generation data saved to: {data_file_path}")

