"""Language model integration for generating reward function feedback.

This module provides functionality to integrate with language models for parsing
feedback and generating reward functions in multi-agent environments.

Security Note: API key usage has been sanitized for open source release.
"""

from openai import OpenAI
import os
from math import sqrt
import json
import numpy as np
from typing import List, Dict, Tuple
from prompt import generate_prompts, TASK_DESCRIPTION_A, TASK_DESCRIPTION_BC

# Initialize OpenAI client - requires OPENAI_API_KEY environment variable
client = None
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        print("Warning: OPENAI_API_KEY environment variable not set")
except Exception as e:
    print(f"Warning: Failed to initialize OpenAI client: {e}")

def get_gpt4_response(prompt, map_type):
    """Generate GPT-4 response for given prompt and map type.
    
    Args:
        prompt (str): The input prompt for the language model
        map_type (str): The map type ("A", "B", or "C")
        
    Returns:
        str: The response from GPT-4, or None if error occurs
    """
    if not client:
        print("Error: OpenAI client not initialized. Please set OPENAI_API_KEY environment variable.")
        return None
        
    if map_type == "A":
        sys_prompt = TASK_DESCRIPTION_A
    elif map_type == "B" or map_type == "C":
        sys_prompt = TASK_DESCRIPTION_BC
    else:
        print(f"Warning: Unknown map type {map_type}, using default")
        sys_prompt = TASK_DESCRIPTION_A
        
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON. " + sys_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


def test_feedback_parsing(num_agents, feedback, map_type="A"):
    """Parse feedback using language model.
    
    Args:
        num_agents (int): Number of agents
        feedback (str): Feedback text to parse
        map_type (str): Map type for context
        
    Returns:
        str: Parsed feedback response or None if error
    """
    prompts = generate_prompts(num_agents, feedback)
    return get_gpt4_response(prompts["feedback_parsing"], map_type)

def test_reward_function_build(num_agents, feedback, map_type="A"):
    """Build reward functions from feedback using language model.
    
    Args:
        num_agents (int): Number of agents
        feedback (dict): Feedback dictionary with agent-specific feedback
        map_type (str): Map type for context
        
    Returns:
        dict: Agent responses with reward functions
    """
    prompts = generate_prompts(0, "")
    
    # Initialize a dictionary to store responses for each agent
    agent_responses = {i: [] for i in range(num_agents)}

    for key, value in feedback.items():
        response = get_gpt4_response(prompts["reward_function_build"].format(feedback=value), map_type)
        
        if response is None:
            print(f"Skipping {key} due to API error")
            continue

        try:
            parsed_response = json.loads(response)
            # Add the original feedback to the parsed response
            parsed_response['original_feedback'] = value
            
            if key.lower() == 'all':
                # If the key is 'all', add the response to all agents
                for agent in agent_responses:
                    agent_responses[agent].append(parsed_response)
            else:
                # Try to extract agent number from the key, if possible
                try:
                    agent_num = int(key.split('_')[-1])
                    if agent_num < num_agents:
                        agent_responses[agent_num].append(parsed_response)
                    else:
                        print(f"Warning: Agent {agent_num} is out of range. Skipping.")
                except ValueError:
                    print(f"Warning: Could not parse agent number from '{key}'. Skipping.")

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response for {key}: {e}")

    return agent_responses

from math import sqrt

import ast
import re

def parse_reward_function(reward_function_str):
    """Parse and validate reward function string safely.
    
    Args:
        reward_function_str (str): String representation of lambda function
        
    Returns:
        callable: Parsed reward function or default function if error
    """
    # Remove any newlines, comments and extra spaces
    reward_function_str = re.sub(r'#.*', '', reward_function_str)
    reward_function_str = re.sub(r'\s+', ' ', reward_function_str.strip())
    
    try:
        # Parse the input string as an expression
        expr_ast = ast.parse(reward_function_str, mode='eval')
        lambda_func = expr_ast.body

        # Ensure it's a lambda function
        if not isinstance(lambda_func, ast.Lambda):
            raise ValueError("The expression is not a lambda function.")

        # Check the number of arguments in the lambda function
        arg_count = len(lambda_func.args.args)

        # Safely compile and evaluate the lambda function with restricted globals
        # Only allow safe mathematical operations
        safe_globals = {
            '__builtins__': {},
            'sqrt': sqrt,
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len
        }
        code = compile(expr_ast, '<string>', 'eval')
        func = eval(code, safe_globals, {})

        # Wrap the function to accept 't' if necessary
        if arg_count == 3:
            # Function already accepts 'obs, act, t'
            return func
        elif arg_count == 2:
            # Function accepts 'obs, act'; add 't' as an extra parameter
            return lambda obs, act, t: func(obs, act)
        else:
            print(f"Lambda function has unexpected number of arguments: {arg_count}")
            return lambda obs, act, t: 0  # Return a default function

    except Exception as e:
        print(f"Error parsing reward function: {e}")
        return lambda obs, act, t: 0  # Return a default function

def reward_aggregation(reward_functions):
    """Aggregate multiple reward functions into a single function.
    
    Args:
        reward_functions (list): List of reward function dictionaries
        
    Returns:
        tuple: (aggregated_function, function_string_representation)
    """
    if not reward_functions or all('reward_function' not in rf for rf in reward_functions):
        return lambda obs, act, ori_reward, t, weights=None: ori_reward, ""

    # Note: Direct eval() usage has been replaced with safer AST parsing

    parsed_reward_functions = []
    for rf in reward_functions:
        if 'reward_function' in rf:
            while True:
                try:
                    parsed_func = parse_reward_function(rf['reward_function'])
                    parsed_reward_functions.append(parsed_func)
                    break  # If successful, exit the loop
                except Exception as e:
                    print(f"Error parsing reward function: {e}")
                    print(f"Problematic function: {rf['reward_function']}")
                    print("Retrying...")
                    # You might want to add a small delay here to avoid tight loops
                    import time
                    time.sleep(1)

    if not parsed_reward_functions:
        return lambda obs, act, ori_reward, t, weights=None: ori_reward, ""


    default_weights = [1.0] * len(parsed_reward_functions)

    def shaped_reward(obs, act, ori_reward, t, weights=None):
        if weights is None:
            weights = [1.0] * len(parsed_reward_functions)
        total_reward = 0 # ori_reward
        for i, reward_function in enumerate(parsed_reward_functions):
            function_reward = reward_function(obs, act, t)
            total_reward += function_reward * weights[i]
        return total_reward

    shaped_reward_str = f"""
        from math import sqrt

        def shaped_reward(obs, act, ori_reward, t, weights=None):
            if weights is None:
                weights = {default_weights}
            total_reward = 0
            reward_functions = [
        {chr(10).join(['        ' + rf['reward_function'] for rf in reward_functions if 'reward_function' in rf])}
            ]
            for i, reward_function in enumerate(reward_functions):
                function_reward = reward_function(obs, act, t)
                total_reward += function_reward * weights[i]
            return total_reward
        """

    return shaped_reward, shaped_reward_str


def assess_feedback_quality(feedback_text: str, parsed_feedback: Dict, performance_history: List[float] = None) -> float:
    """
    Assess the quality of human feedback based on multiple criteria.
    
    Args:
        feedback_text (str): Original feedback text
        parsed_feedback (Dict): Parsed feedback dictionary
        performance_history (List[float], optional): Performance scores from previous generations
        
    Returns:
        float: Quality score between 0.0 and 1.0
    """
    quality_score = 0.0
    weight_sum = 0.0
    
    # 1. Length and informativeness (weight: 0.25)
    length_score = min(1.0, len(feedback_text.split()) / 20.0)  # Optimal around 20 words
    quality_score += 0.25 * length_score
    weight_sum += 0.25
    
    # 2. Specificity - contains specific agent mentions or actions (weight: 0.30)
    specificity_indicators = ['agent', 'green', 'rose', 'pink', 'red', 'blue', 'chop', 'deliver', 
                             'tomato', 'lettuce', 'onion', 'plate', 'knife', 'cutting', 'board']
    specificity_count = sum(1 for indicator in specificity_indicators if indicator.lower() in feedback_text.lower())
    specificity_score = min(1.0, specificity_count / 5.0)  # Optimal around 5 specific terms
    quality_score += 0.30 * specificity_score
    weight_sum += 0.30
    
    # 3. Actionability - contains actionable suggestions (weight: 0.25)
    actionable_indicators = ['should', 'need', 'must', 'try', 'focus', 'avoid', 'improve', 'better', 'faster', 'coordinate']
    actionable_count = sum(1 for indicator in actionable_indicators if indicator.lower() in feedback_text.lower())
    actionable_score = min(1.0, actionable_count / 3.0)  # Optimal around 3 actionable terms
    quality_score += 0.25 * actionable_score
    weight_sum += 0.25
    
    # 4. Consistency with performance trends (weight: 0.20)
    if performance_history and len(performance_history) >= 2:
        recent_trend = performance_history[-1] - performance_history[-2]
        # Check if feedback sentiment aligns with performance trend
        negative_indicators = ['slow', 'bad', 'wrong', 'avoid', 'stop', 'not', 'problem', 'issue']
        positive_indicators = ['good', 'great', 'better', 'excellent', 'perfect', 'nice', 'well']
        
        negative_count = sum(1 for indicator in negative_indicators if indicator.lower() in feedback_text.lower())
        positive_count = sum(1 for indicator in positive_indicators if indicator.lower() in feedback_text.lower())
        
        sentiment_score = (positive_count - negative_count) / max(1, positive_count + negative_count)
        
        # Check alignment: negative sentiment should align with negative trend and vice versa
        alignment_score = 0.5  # Neutral baseline
        if (recent_trend < 0 and sentiment_score < 0) or (recent_trend > 0 and sentiment_score > 0):
            alignment_score = 1.0  # Good alignment
        elif (recent_trend < 0 and sentiment_score > 0) or (recent_trend > 0 and sentiment_score < 0):
            alignment_score = 0.2  # Poor alignment
            
        quality_score += 0.20 * alignment_score
        weight_sum += 0.20
    
    # Normalize by total weight
    return quality_score / weight_sum if weight_sum > 0 else 0.5


def assess_reward_function_validity(reward_function_str: str, obs_sample: np.ndarray = None) -> Tuple[bool, float]:
    """
    Assess the validity and potential effectiveness of a generated reward function.
    
    Args:
        reward_function_str (str): String representation of the reward function
        obs_sample (np.ndarray, optional): Sample observation for testing
        
    Returns:
        tuple: (is_valid, effectiveness_score)
    """
    try:
        # Test if the function can be parsed
        parsed_func = parse_reward_function(reward_function_str)
        
        # Test with sample observation if provided
        if obs_sample is not None:
            test_obs = obs_sample if isinstance(obs_sample, np.ndarray) else np.random.rand(32)
            test_act = 1
            test_t = 10
            
            # Try to execute the function
            reward = parsed_func(test_obs, test_act, test_t)
            
            # Check if reward is reasonable (not NaN, not infinite, within bounds)
            if np.isnan(reward) or np.isinf(reward) or abs(reward) > 1000:
                return False, 0.0
            
            # Effectiveness heuristics
            effectiveness_score = 0.5  # Base score
            
            # Check if function uses observation data (not constant)
            test_obs_2 = np.random.rand(32)
            reward_2 = parsed_func(test_obs_2, test_act, test_t)
            
            if abs(reward - reward_2) > 1e-6:  # Function varies with observation
                effectiveness_score += 0.3
                
            # Check if function uses action data
            reward_3 = parsed_func(test_obs, test_act + 1, test_t)
            if abs(reward - reward_3) > 1e-6:  # Function varies with action
                effectiveness_score += 0.2
                
            return True, effectiveness_score
        
        return True, 0.5  # Valid but unknown effectiveness
        
    except Exception as e:
        return False, 0.0


def filter_low_quality_feedback(feedback_list: List[Dict], quality_threshold: float = 0.4) -> List[Dict]:
    """
    Filter out low-quality feedback based on assessment scores.
    
    Args:
        feedback_list (List[Dict]): List of feedback dictionaries
        quality_threshold (float): Minimum quality score to keep feedback
        
    Returns:
        List[Dict]: Filtered feedback list with quality scores added
    """
    filtered_feedback = []
    
    for feedback_dict in feedback_list:
        if 'original_feedback' in feedback_dict:
            quality_score = assess_feedback_quality(feedback_dict['original_feedback'], feedback_dict)
            feedback_dict['quality_score'] = quality_score
            
            if quality_score >= quality_threshold:
                filtered_feedback.append(feedback_dict)
            else:
                print(f"Filtered out low-quality feedback (score: {quality_score:.2f}): {feedback_dict['original_feedback'][:100]}...")
    
    return filtered_feedback


    
def single_reward_function(reward_function_dict):
    """Create single reward function from dictionary.
    
    Args:
        reward_function_dict (dict): Dictionary containing reward function info
        
    Returns:
        tuple: (reward_function, reward_function_string)
    """
    def parse_reward_function_safe(reward_function_str):
        """Safely parse reward function string."""
        # This is a simplified version - in production, use AST parsing
        body = reward_function_str.replace("lambda obs, act:", "").strip()
        safe_globals = {
            '__builtins__': {},
            'sqrt': sqrt,
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len
        }
        return eval(f"lambda obs, act, ori_reward, t: {body}", safe_globals, {})

    parsed_function = parse_reward_function_safe(reward_function_dict['reward_function'])

    def shaped_reward(obs, act, ori_reward, t):
        return parsed_function(obs, act, ori_reward, t)

    shaped_reward_str = f"""
    from math import sqrt

    def shaped_reward(obs, act, ori_reward, t):
        return {reward_function_dict['reward_function']}

    # Explanation: {reward_function_dict['explanation']}
    # Original feedback: {reward_function_dict['original_feedback']}
    """

    return shaped_reward, shaped_reward_str
