#!/usr/bin/env python3
"""
Baseline Comparison Script

This script runs M3HF against baseline algorithms (IPPO, MAPPO) and 
compares their performance on the Overcooked environment.
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from m3hf import M3HFTrainer
from utils import setup_logging

def create_baseline_config(algorithm='IPPO'):
    """Create baseline algorithm configuration."""
    return {
        # Environment settings
        'env_id': 'Overcooked-MA-v1',
        'n_agent': 3,
        'grid_dim': [7, 7],
        'map_type': 'A',
        'task': 6,  # Most complex task
        'obs_radius': 2,
        'mode': 'vector',
        
        # Training settings
        'algorithm': algorithm,
        'training_iterations': 1000,
        'learning_rate': 3e-4,
        'batch_size': 5120,
        'epochs': 10,
        'gamma': 0.99,
        
        # Evaluation
        'eval_frequency': 50,
        'eval_episodes': 10,
        
        # Logging
        'use_wandb': False,
        'verbose': True,
    }

def create_m3hf_config():
    """Create M3HF configuration."""
    config = create_baseline_config('IPPO')  # M3HF uses IPPO as backbone
    config.update({
        # M3HF specific settings
        'generations': 5,
        'training_iterations': 200,  # Per generation
        'feedback_frequency': 50,
        'weight_decay_factor': 0.95,
        'performance_adjustment_factor': 0.1,
        'quality_threshold': 0.3,
        'similarity_threshold': 0.8,
    })
    return config

def simple_feedback_generator(generation, rollout_data):
    """Generate feedback for M3HF based on generation."""
    feedback_by_generation = {
        0: "Agents need better coordination. The red agent should focus on vegetables, blue agent on plates, and green agent on cutting.",
        1: "Good progress! Now focus on timing - don't let ingredients sit idle. Move faster between tasks.",
        2: "Agents are improving. Make sure to properly chop all vegetables before combining on plates.",
        3: "Excellent coordination! Now optimize delivery timing - make sure salads are complete before delivering.",
        4: "Perfect teamwork! Focus on speed and efficiency while maintaining accuracy.",
    }
    return feedback_by_generation.get(generation, "Keep improving coordination and speed!")

def run_baseline_experiment(config, output_dir, seed=42):
    """Run a single baseline experiment."""
    logger = logging.getLogger(__name__)
    algorithm = config['algorithm']
    
    logger.info(f"Starting {algorithm} experiment with seed {seed}")
    
    # Set random seeds
    np.random.seed(seed)
    
    # Create output directory
    exp_dir = os.path.join(output_dir, f"{algorithm.lower()}_seed_{seed}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    start_time = time.time()
    
    try:
        if algorithm in ['IPPO', 'MAPPO']:
            # Run baseline with Ray RLLib
            from play_rllib_ippo import train_baseline
            results = train_baseline(config, exp_dir)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        duration = time.time() - start_time
        
        # Save results
        results['duration'] = duration
        results['seed'] = seed
        results['algorithm'] = algorithm
        
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"{algorithm} experiment completed in {duration:.1f}s")
        return results
        
    except Exception as e:
        logger.error(f"{algorithm} experiment failed: {e}")
        return {'error': str(e), 'algorithm': algorithm, 'seed': seed}

def run_m3hf_experiment(config, output_dir, seed=42):
    """Run M3HF experiment."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting M3HF experiment with seed {seed}")
    
    # Set random seeds
    np.random.seed(seed)
    
    # Create output directory
    exp_dir = os.path.join(output_dir, f"m3hf_seed_{seed}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    start_time = time.time()
    
    try:
        # Initialize trainer
        trainer = M3HFTrainer(config)
        trainer.set_feedback_generator(simple_feedback_generator)
        
        # Run training
        results = trainer.train()
        
        duration = time.time() - start_time
        
        # Save results
        results['duration'] = duration
        results['seed'] = seed
        results['algorithm'] = 'M3HF'
        
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"M3HF experiment completed in {duration:.1f}s")
        return results
        
    except Exception as e:
        logger.error(f"M3HF experiment failed: {e}")
        return {'error': str(e), 'algorithm': 'M3HF', 'seed': seed}

def plot_comparison_results(results, output_dir):
    """Plot comparison results."""
    logger = logging.getLogger(__name__)
    logger.info("Plotting comparison results...")
    
    # Organize results by algorithm
    algorithm_results = {}
    for result in results:
        if 'error' in result:
            continue
            
        algorithm = result['algorithm']
        if algorithm not in algorithm_results:
            algorithm_results[algorithm] = []
        algorithm_results[algorithm].append(result)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Baseline Comparison Results', fontsize=16)
    
    # Plot 1: Final Performance Comparison
    ax = axes[0, 0]
    algorithms = list(algorithm_results.keys())
    final_rewards = []
    final_rewards_std = []
    
    for algorithm in algorithms:
        rewards = [r.get('final_performance', 0) for r in algorithm_results[algorithm]]
        final_rewards.append(np.mean(rewards))
        final_rewards_std.append(np.std(rewards))
    
    bars = ax.bar(algorithms, final_rewards, yerr=final_rewards_std, capsize=5)
    ax.set_title('Final Performance Comparison')
    ax.set_ylabel('Average Episode Return')
    
    # Add value labels on bars
    for bar, reward in zip(bars, final_rewards):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{reward:.1f}', ha='center', va='bottom')
    
    # Plot 2: Learning Curves (if available)
    ax = axes[0, 1]
    for algorithm in algorithms:
        for i, result in enumerate(algorithm_results[algorithm]):
            learning_curve = result.get('learning_curve', [])
            if learning_curve:
                alpha = 0.3 if i > 0 else 1.0  # First curve opaque, others transparent
                ax.plot(learning_curve, label=f'{algorithm}' if i == 0 else "", alpha=alpha)
    
    ax.set_title('Learning Curves')
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Episode Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Training Duration
    ax = axes[1, 0]
    durations = [np.mean([r.get('duration', 0) for r in algorithm_results[alg]]) 
                for alg in algorithms]
    duration_std = [np.std([r.get('duration', 0) for r in algorithm_results[alg]]) 
                   for alg in algorithms]
    
    bars = ax.bar(algorithms, durations, yerr=duration_std, capsize=5)
    ax.set_title('Training Duration')
    ax.set_ylabel('Time (seconds)')
    
    # Add value labels
    for bar, duration in zip(bars, durations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{duration:.0f}s', ha='center', va='bottom')
    
    # Plot 4: Sample Efficiency (if available)
    ax = axes[1, 1]
    for algorithm in algorithms:
        sample_efficiency = []
        for result in algorithm_results[algorithm]:
            # Calculate sample efficiency as performance per training step
            final_perf = result.get('final_performance', 0)
            total_steps = result.get('total_training_steps', 1)
            sample_efficiency.append(final_perf / max(total_steps, 1) * 1000)  # per 1K steps
        
        if sample_efficiency:
            ax.bar(algorithm, np.mean(sample_efficiency), 
                  yerr=np.std(sample_efficiency), capsize=5)
    
    ax.set_title('Sample Efficiency')
    ax.set_ylabel('Performance per 1K Training Steps')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'comparison_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Comparison plot saved to {plot_path}")
    
    plt.show()

def main():
    """Run baseline comparison experiments."""
    parser = argparse.ArgumentParser(description='Baseline Comparison Experiments')
    parser.add_argument('--algorithms', nargs='+', default=['IPPO', 'MAPPO', 'M3HF'],
                       help='Algorithms to compare')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                       help='Random seeds for experiments')
    parser.add_argument('--output_dir', type=str, default='results/baseline_comparison',
                       help='Output directory')
    parser.add_argument('--parallel', action='store_true',
                       help='Run experiments in parallel')
    parser.add_argument('--plot_only', action='store_true',
                       help='Only plot results from existing experiments')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting baseline comparison experiments")
    logger.info(f"Algorithms: {args.algorithms}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Output directory: {args.output_dir}")
    
    if not args.plot_only:
        # Run experiments
        all_results = []
        
        if args.parallel:
            # Run experiments in parallel
            with ProcessPoolExecutor() as executor:
                futures = []
                
                for algorithm in args.algorithms:
                    for seed in args.seeds:
                        if algorithm == 'M3HF':
                            config = create_m3hf_config()
                            future = executor.submit(run_m3hf_experiment, config, 
                                                   args.output_dir, seed)
                        else:
                            config = create_baseline_config(algorithm)
                            future = executor.submit(run_baseline_experiment, config, 
                                                   args.output_dir, seed)
                        futures.append(future)
                
                # Collect results
                for future in futures:
                    result = future.result()
                    all_results.append(result)
        else:
            # Run experiments sequentially
            for algorithm in args.algorithms:
                for seed in args.seeds:
                    if algorithm == 'M3HF':
                        config = create_m3hf_config()
                        result = run_m3hf_experiment(config, args.output_dir, seed)
                    else:
                        config = create_baseline_config(algorithm)
                        result = run_baseline_experiment(config, args.output_dir, seed)
                    all_results.append(result)
        
        # Save combined results
        with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
    
    else:
        # Load existing results for plotting
        results_file = os.path.join(args.output_dir, 'all_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                all_results = json.load(f)
        else:
            logger.error(f"Results file not found: {results_file}")
            return
    
    # Plot results
    plot_comparison_results(all_results, args.output_dir)
    
    # Print summary
    logger.info("\nExperiment Summary:")
    for algorithm in args.algorithms:
        alg_results = [r for r in all_results if r.get('algorithm') == algorithm and 'error' not in r]
        if alg_results:
            final_perfs = [r.get('final_performance', 0) for r in alg_results]
            logger.info(f"{algorithm}: {np.mean(final_perfs):.2f} ± {np.std(final_perfs):.2f}")
        else:
            logger.info(f"{algorithm}: No successful runs")

if __name__ == '__main__':
    main()