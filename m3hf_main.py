"""M3HF Main Training Script.

This script provides a main entry point for training multi-agent PPO policies
on the Overcooked environment and generating rollout videos.

Usage:
    python m3hf_main.py [OPTIONS]
    
Example:
    python m3hf_main.py --num_workers 16 --num_gpus 2 --training_iterations 200
"""

import argparse
import os
import time
import ray
from play_rllib_ippo import train_multi_agent_ppo
from rollout_lstm import render_video_from_checkpoint


def main(env_id, num_workers, num_gpus, training_iterations, checkpoint_interval, video_dir):
    """Main training function.
    
    Args:
        env_id (str): Environment identifier
        num_workers (int): Number of rollout workers
        num_gpus (int): Number of GPUs to use
        training_iterations (int): Total training iterations
        checkpoint_interval (int): Interval for checkpoints
        video_dir (str): Directory to save videos
    """

    env_config = {
        'env_id': env_id,
        'grid_dim': [7, 7],
        'task': "lettuce-onion-tomato salad",
        'rewardList': {"subtask finished": 10, "correct delivery": 200, "wrong delivery": -5, "step penalty": -0.1},
        'map_type': "A",
        'n_agent': 3,
        'obs_radius': 2,
        'mode': "vector",
        'debug': False
    }


    # Initialize Ray with error handling
    try:
        ray.init(ignore_reinit_error=True, num_cpus=num_workers * 2, num_gpus=num_gpus)
        print(f"Ray initialized with {num_workers * 2} CPUs and {num_gpus} GPUs")
    except Exception as e:
        print(f"Error initializing Ray: {e}")
        return


    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)


    os.makedirs(video_dir, exist_ok=True)


    # Training loop with checkpoint intervals
    for i in range(0, training_iterations, checkpoint_interval):
        print(f"Starting training iteration {i} to {i + checkpoint_interval}")
        
        try:
            train_multi_agent_ppo(
                env_config, 
                num_workers, 
                num_gpus, 
                checkpoint_interval, 
                use_wandb=False, 
                wandb_project_name="overcooked_multi_agent"
            )
        except Exception as e:
            print(f"Error during training: {e}")
            continue

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{i + checkpoint_interval}")
        
        # Generate rollout video
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        video_path = os.path.join(video_dir, f"rollout_{timestamp}.mp4")
        
        try:
            render_video_from_checkpoint(checkpoint_path, env_config, video_path)
            print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Error generating video: {e}")

    # Clean shutdown
    try:
        ray.shutdown()
        print("Ray shutdown completed")
    except Exception as e:
        print(f"Error during Ray shutdown: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train M3HF multi-agent policies')
    parser.add_argument('--env_id', type=str, default='Overcooked-MA-v1', 
                       help='Environment ID')
    parser.add_argument('--num_workers', type=int, default=16, 
                       help='Number of rollout workers (default: 16)')
    parser.add_argument('--num_gpus', type=int, default=1, 
                       help='Number of GPUs to use (default: 1)')
    parser.add_argument('--training_iterations', type=int, default=100, 
                       help='Number of training iterations')
    parser.add_argument('--checkpoint_interval', type=int, default=20, 
                       help='Interval for saving checkpoints and generating rollouts')
    parser.add_argument('--video_dir', type=str, default='videos', 
                       help='Directory to save rollout videos')
    
    args = parser.parse_args()

    main(
        args.env_id, 
        args.num_workers, 
        args.num_gpus, 
        args.training_iterations, 
        args.checkpoint_interval, 
        args.video_dir
    )
