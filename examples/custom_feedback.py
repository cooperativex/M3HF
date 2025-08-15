#!/usr/bin/env python3
"""
Custom Feedback Integration Example

This script demonstrates how to integrate custom human feedback
into the M3HF training process, including different feedback
types and quality levels.
"""

import sys
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from m3hf import M3HFTrainer
from language import assess_feedback_quality, compute_sentence_similarity

class CustomFeedbackGenerator:
    """
    Custom feedback generator that simulates different types of human feedback
    including expert feedback, novice feedback, and mixed-quality feedback.
    """
    
    def __init__(self, feedback_style='mixed'):
        """
        Initialize feedback generator.
        
        Args:
            feedback_style (str): 'expert', 'novice', or 'mixed'
        """
        self.feedback_style = feedback_style
        self.generation_count = 0
        self.previous_feedback = []
        
        # Expert feedback templates (high quality, specific)
        self.expert_feedback = {
            'coordination': [
                "Agent 1 should specialize in vegetable collection while Agent 2 handles plate preparation and Agent 3 manages cutting stations.",
                "Implement role-based coordination: designate one agent as the primary chopper to avoid conflicts at cutting stations.",
                "Use spatial coordination: agents should work in different areas to minimize collision and maximize parallel processing."
            ],
            'efficiency': [
                "Optimize task sequencing: collect all ingredients before starting any chopping to reduce idle time.",
                "Implement pipeline processing: while one agent chops, others should prepare the next batch of ingredients.",
                "Reduce unnecessary movements by planning ingredient collection routes more efficiently."
            ],
            'specific_errors': [
                "Agent 2 repeatedly tries to chop unchopped vegetables - ensure proper ingredient preparation sequence.",
                "Agents are delivering incomplete salads - verify all required ingredients are properly chopped and combined.",
                "Multiple agents competing for the same knife - implement turn-taking or role assignment."
            ]
        }
        
        # Novice feedback templates (lower quality, less specific)
        self.novice_feedback = {
            'general': [
                "The agents should work better together.",
                "They need to be faster and more efficient.", 
                "Something is wrong with their coordination.",
                "They keep making mistakes with the food preparation."
            ],
            'vague': [
                "The red agent is not doing well.",
                "The teamwork could be improved somehow.",
                "They should try a different approach.",
                "The performance is not satisfactory."
            ],
            'contradictory': [
                "All agents should focus on the same task simultaneously.",
                "Agents should never use the cutting stations.",
                "Speed is more important than accuracy - deliver anything quickly."
            ]
        }
    
    def generate_expert_feedback(self, generation: int, rollout_data: Dict) -> str:
        """Generate high-quality expert feedback."""
        performance = rollout_data.get('mean_reward', 0)
        
        if generation == 0:
            # Initial coordination advice
            category = 'coordination'
        elif performance < -50:
            # Focus on specific errors
            category = 'specific_errors'
        else:
            # Focus on efficiency improvements
            category = 'efficiency'
        
        feedback = random.choice(self.expert_feedback[category])
        
        # Add performance-specific details
        if performance < -100:
            feedback += " Current performance is very low - focus on basic task completion first."
        elif performance < 0:
            feedback += " Performance is improving but still needs coordination refinement."
        else:
            feedback += " Good progress - now optimize for speed and efficiency."
        
        return feedback
    
    def generate_novice_feedback(self, generation: int, rollout_data: Dict) -> str:
        """Generate lower-quality novice feedback."""
        feedback_types = list(self.novice_feedback.keys())
        
        # Novice feedback is more random and less targeted
        category = random.choice(feedback_types)
        feedback = random.choice(self.novice_feedback[category])
        
        # Sometimes add unhelpful details
        if random.random() < 0.3:
            feedback += " Maybe try doing things differently."
        
        return feedback
    
    def generate_mixed_quality_feedback(self, generation: int, rollout_data: Dict) -> str:
        """Generate mixed quality feedback (expert + novice)."""
        # 60% expert, 40% novice feedback
        if random.random() < 0.6:
            return self.generate_expert_feedback(generation, rollout_data)
        else:
            return self.generate_novice_feedback(generation, rollout_data)
    
    def __call__(self, generation: int, rollout_data: Dict) -> str:
        """Generate feedback based on the configured style."""
        self.generation_count += 1
        
        if self.feedback_style == 'expert':
            feedback = self.generate_expert_feedback(generation, rollout_data)
        elif self.feedback_style == 'novice':
            feedback = self.generate_novice_feedback(generation, rollout_data)
        else:  # mixed
            feedback = self.generate_mixed_quality_feedback(generation, rollout_data)
        
        # Store for similarity checking
        self.previous_feedback.append(feedback)
        
        return feedback

def analyze_feedback_quality(feedback_list: List[str]):
    """Analyze the quality of a list of feedback messages."""
    print("\n=== Feedback Quality Analysis ===")
    
    quality_scores = []
    for i, feedback in enumerate(feedback_list):
        quality = assess_feedback_quality(feedback)
        quality_scores.append(quality)
        print(f"Feedback {i}: Quality = {quality:.3f}")
        print(f"  Text: {feedback[:80]}...")
        print()
    
    print(f"Average Quality Score: {np.mean(quality_scores):.3f}")
    print(f"Quality Standard Deviation: {np.std(quality_scores):.3f}")
    
    return quality_scores

def analyze_feedback_similarity(feedback_list: List[str]):
    """Analyze similarity between feedback messages."""
    print("\n=== Feedback Similarity Analysis ===")
    
    similarities = []
    for i in range(len(feedback_list)):
        for j in range(i+1, len(feedback_list)):
            similarity = compute_sentence_similarity(feedback_list[i], feedback_list[j])
            similarities.append(similarity)
            print(f"Feedback {i} vs {j}: Similarity = {similarity:.3f}")
    
    if similarities:
        print(f"Average Similarity: {np.mean(similarities):.3f}")
        print(f"Max Similarity: {np.max(similarities):.3f}")
    
    return similarities

def demo_feedback_generation():
    """Demo different types of feedback generation."""
    print("=== Feedback Generation Demo ===")
    
    # Sample rollout data
    sample_rollout_data = {
        'mean_reward': -25.5,
        'episode_length': 150,
        'success_rate': 0.2
    }
    
    feedback_styles = ['expert', 'novice', 'mixed']
    
    for style in feedback_styles:
        print(f"\n--- {style.title()} Feedback Style ---")
        generator = CustomFeedbackGenerator(style)
        
        feedback_samples = []
        for gen in range(3):
            feedback = generator(gen, sample_rollout_data)
            feedback_samples.append(feedback)
            print(f"Generation {gen}: {feedback}")
        
        # Analyze quality
        analyze_feedback_quality(feedback_samples)

def demo_feedback_processing():
    """Demo feedback processing pipeline."""
    print("\n=== Feedback Processing Demo ===")
    
    # Generate sample feedback
    generator = CustomFeedbackGenerator('mixed')
    feedback_samples = []
    
    for gen in range(5):
        rollout_data = {
            'mean_reward': random.uniform(-100, 50),
            'episode_length': random.randint(100, 200),
            'success_rate': random.uniform(0, 0.8)
        }
        feedback = generator(gen, rollout_data)
        feedback_samples.append(feedback)
    
    print("Generated feedback samples:")
    for i, feedback in enumerate(feedback_samples):
        print(f"{i}: {feedback}")
    
    # Analyze quality and similarity
    quality_scores = analyze_feedback_quality(feedback_samples)
    similarity_scores = analyze_feedback_similarity(feedback_samples)
    
    # Filter low-quality feedback
    quality_threshold = 0.3
    filtered_feedback = [
        feedback for feedback, quality in zip(feedback_samples, quality_scores)
        if quality >= quality_threshold
    ]
    
    print(f"\nFiltered feedback (quality >= {quality_threshold}):")
    print(f"Original count: {len(feedback_samples)}")
    print(f"Filtered count: {len(filtered_feedback)}")
    for i, feedback in enumerate(filtered_feedback):
        print(f"{i}: {feedback}")

def run_training_with_custom_feedback():
    """Run M3HF training with custom feedback generator."""
    print("\n=== M3HF Training with Custom Feedback ===")
    
    # Create M3HF configuration
    config = {
        'env_id': 'Overcooked-MA-v1',
        'n_agent': 3,
        'map_type': 'A',
        'task': 3,
        'generations': 3,
        'training_iterations': 100,  # Keep short for demo
        'feedback_frequency': 25,
        'verbose': True,
        'use_wandb': False,
    }
    
    # Test different feedback styles
    feedback_styles = ['expert', 'novice', 'mixed']
    results = {}
    
    for style in feedback_styles:
        print(f"\n--- Training with {style} feedback ---")
        
        # Create feedback generator
        feedback_generator = CustomFeedbackGenerator(style)
        
        # Initialize trainer
        trainer = M3HFTrainer(config)
        trainer.set_feedback_generator(feedback_generator)
        
        try:
            # Run training
            result = trainer.train()
            results[style] = result
            
            print(f"Final performance with {style} feedback: {result.get('final_performance', 'N/A')}")
            
        except Exception as e:
            print(f"Training with {style} feedback failed: {e}")
            results[style] = {'error': str(e)}
    
    # Compare results
    print("\n=== Results Comparison ===")
    for style, result in results.items():
        if 'error' not in result:
            final_perf = result.get('final_performance', 0)
            print(f"{style.title()} feedback: {final_perf:.2f}")
        else:
            print(f"{style.title()} feedback: Failed - {result['error']}")
    
    return results

def main():
    """Run custom feedback integration demos."""
    print("Custom Feedback Integration Demo")
    print("=" * 50)
    
    # Run all demos
    demo_feedback_generation()
    demo_feedback_processing()
    
    # Optional: Run training demo (requires more time)
    run_training = input("\nRun training demo? (y/n): ").lower().startswith('y')
    if run_training:
        run_training_with_custom_feedback()
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nKey takeaways:")
    print("1. Expert feedback leads to better performance")
    print("2. Quality assessment helps filter bad feedback")
    print("3. Similarity checking prevents redundant feedback")
    print("4. Mixed feedback requires robust handling")

if __name__ == '__main__':
    main()