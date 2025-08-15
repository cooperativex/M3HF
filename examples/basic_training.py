#!/usr/bin/env python3
"""
Basic M3HF Training Example

This script demonstrates how to run a basic M3HF training session
with simulated human feedback on the Overcooked environment.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from m3hf import M3HFTrainer
from utils import setup_logging

def create_m3hf_config():
    """Create a basic M3HF configuration."""
    return {
        # Environment settings
        'env_id': 'Overcooked-MA-v1',
        'n_agent': 3,
        'grid_dim': [7, 7],
        'map_type': 'A',  # Start with easiest map
        'task': 3,        # Lettuce-tomato salad (medium complexity)
        'obs_radius': 2,
        'mode': 'vector',
        
        # M3HF Algorithm settings
        'generations': 3,  # Keep short for example
        'training_iterations': 200,
        'episodes_per_iteration': 10,
        'rollout_episodes': 3,
        'feedback_frequency': 50,  # Get feedback every 50 iterations
        
        # Weight adjustment settings
        'weight_decay_factor': 0.95,
        'performance_adjustment_factor': 0.1,
        'quality_threshold': 0.3,
        'similarity_threshold': 0.8,
        
        # Training settings
        'learning_rate': 3e-4,
        'batch_size': 1024,
        'epochs': 5,
        'gamma': 0.99,
        
        # Logging
        'use_wandb': False,  # Disable for basic example
        'verbose': True,
    }

def example_feedback_generator(generation, rollout_data):
    """
    Simple example feedback generator that provides different feedback
    for each generation.
    
    Args:
        generation (int): Current generation number
        rollout_data (dict): Rollout statistics and info
        
    Returns:
        str: Human feedback message
    """
    feedback_templates = {
        0: "The agents need to work together better. The red agent should focus on getting vegetables while the other agents prepare plates and cutting stations.",
        1: "Good improvement! Now the agents should coordinate timing better - one agent should cut vegetables while others get ready to combine ingredients on plates.", 
        2: "The agents are learning well. Focus on delivering the correct salad quickly - make sure all ingredients are properly chopped before combining.",
    }
    
    return feedback_templates.get(generation, "Keep improving coordination and efficiency!")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Basic M3HF Training Example')
    parser.add_argument('--generations', type=int, default=3, help='Number of training generations')
    parser.add_argument('--training_iterations', type=int, default=200, help='Iterations per generation')
    parser.add_argument('--map_type', type=str, default='A', choices=['A', 'B', 'C'], help='Environment map')
    parser.add_argument('--task', type=int, default=3, help='Task ID (0-6)')
    parser.add_argument('--output_dir', type=str, default='results/basic_example', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Basic M3HF Training Example")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create configuration
    config = create_m3hf_config()
    config['generations'] = args.generations
    config['training_iterations'] = args.training_iterations
    config['map_type'] = args.map_type
    config['task'] = args.task
    config['output_dir'] = args.output_dir
    config['verbose'] = args.verbose
    
    logger.info(f"Configuration: {config}")
    
    try:
        # Initialize trainer
        trainer = M3HFTrainer(config)
        
        # Set custom feedback generator
        trainer.set_feedback_generator(example_feedback_generator)
        
        # Run training
        logger.info("Starting M3HF training...")
        results = trainer.train()
        
        # Print results
        logger.info("Training completed!")
        logger.info(f"Final performance: {results.get('final_performance', 'N/A')}")
        logger.info(f"Results saved to: {args.output_dir}")
        
        # Print generation summary
        for gen in range(config['generations']):
            gen_results = results.get(f'generation_{gen}', {})
            logger.info(f"Generation {gen}: {gen_results.get('mean_reward', 'N/A')} reward")
            
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == '__main__':
    main()