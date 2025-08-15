import numpy as np
from gym.spaces import Box, Discrete
from ray.rllib.utils import try_import_torch
import math

torch, _ = try_import_torch()

def serialize_gym_space(space):
    if isinstance(space, Box):
        return {
            "type": "Box",
            "low": space.low.tolist() if isinstance(space.low, np.ndarray) else space.low,
            "high": space.high.tolist() if isinstance(space.high, np.ndarray) else space.high,
            "shape": space.shape,
            "dtype": str(space.dtype)
        }
    elif isinstance(space, Discrete):
        return {
            "type": "Discrete",
            "n": space.n
        }
    else:
        return str(space)

def serialize_torch_object(obj):
    if torch and isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj

def serialize_for_wandb(obj):
    if isinstance(obj, dict):
        return {k: serialize_for_wandb(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_wandb(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_for_wandb(v) for v in obj)
    elif isinstance(obj, (Box, Discrete)):
        return serialize_gym_space(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif callable(obj):
        return str(obj)
    else:
        return serialize_torch_object(obj)
    
    
    
def extract_metrics_for_wandb(result):
    metrics = {
        "timesteps_total": result["timesteps_total"],
        "episodes_total": result["episodes_total"],
        "training_iteration": result["training_iteration"],
        "time_total_s": result["time_total_s"],
        "num_env_steps_sampled": result["num_env_steps_sampled"],
        "num_env_steps_trained": result["num_env_steps_trained"],
        "num_agent_steps_sampled": result["num_agent_steps_sampled"],
        "num_agent_steps_trained": result["num_agent_steps_trained"],
    }

    for agent in result["info"]["learner"]:
        agent_stats = result["info"]["learner"][agent]["learner_stats"]
        metrics.update({
            f"{agent}_total_loss": agent_stats["total_loss"],
            f"{agent}_policy_loss": agent_stats["policy_loss"],
            f"{agent}_vf_loss": agent_stats["vf_loss"],
            f"{agent}_kl": agent_stats["kl"],
            f"{agent}_entropy": agent_stats["entropy"],
        })

    if not math.isnan(result["episode_reward_mean"]):
        metrics.update({
            "episode_reward_mean": result["episode_reward_mean"],
            "episode_reward_min": result["episode_reward_min"],
            "episode_reward_max": result["episode_reward_max"],
            "episode_len_mean": result["episode_len_mean"],
        })

    return metrics




from ray.rllib.algorithms.callbacks import DefaultCallbacks
import numpy as np

class EpisodeStatsCallback(DefaultCallbacks):
    def on_episode_start(self, worker, base_env, policies, episode, env_index, **kwargs):
        episode.user_data["episode_rewards"] = []
        episode.user_data["episode_lengths"] = []

    def on_episode_step(self, worker, base_env, episode, env_index, **kwargs):
        total_reward = sum(episode.last_info_for().get("rewards", {}).values())
        episode.user_data["episode_rewards"].append(total_reward)
        episode.user_data["episode_lengths"].append(1) 

    def on_episode_end(self, worker, base_env, policies, episode, env_index, **kwargs):
        total_reward = sum(episode.user_data["episode_rewards"])
        total_length = sum(episode.user_data["episode_lengths"])
        print(f"Episode ended. Total reward: {total_reward}, Length: {total_length}")
        episode.custom_metrics["episode_reward"] = total_reward
        episode.custom_metrics["episode_length"] = total_length

    def on_train_result(self, algorithm, result, **kwargs):
        if "custom_metrics" in result:
            rewards = result["custom_metrics"].get("episode_reward", [])
            lengths = result["custom_metrics"].get("episode_length", [])
            if rewards:
                result["episode_reward_mean"] = np.mean(rewards)
                result["episode_reward_min"] = np.min(rewards)
                result["episode_reward_max"] = np.max(rewards)
            if lengths:
                result["episode_len_mean"] = np.mean(lengths)
                
                
                
def cal_similarity(sentence1, sentence2):
    """
    Calculate the similarity between two sentences using cosine similarity.
    Uses multiple approaches: lexical similarity, semantic embeddings, and LLM-based assessment.
    
    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.
    
    Returns:
        float: A similarity score between 0 and 1, where 1 indicates identical sentences.
    """
    import re
    from collections import Counter
    import os
    
    # Preprocessing
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    tokens1 = preprocess_text(sentence1)
    tokens2 = preprocess_text(sentence2)
    
    # 1. Jaccard Similarity (lexical overlap)
    set1, set2 = set(tokens1), set(tokens2)
    jaccard_sim = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0.0
    
    # 2. Cosine Similarity on word counts
    counter1, counter2 = Counter(tokens1), Counter(tokens2)
    all_words = set(counter1.keys()) | set(counter2.keys())
    
    vec1 = [counter1.get(word, 0) for word in all_words]
    vec2 = [counter2.get(word, 0) for word in all_words]
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    
    cosine_sim = dot_product / (mag1 * mag2) if mag1 * mag2 > 0 else 0.0
    
    # 3. Length-normalized edit distance
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]
    
    edit_distance = levenshtein_distance(' '.join(tokens1), ' '.join(tokens2))
    max_len = max(len(' '.join(tokens1)), len(' '.join(tokens2)))
    edit_sim = 1.0 - (edit_distance / max_len) if max_len > 0 else 0.0
    
    # 4. Semantic similarity using simple word overlap with weights
    # Weight important words higher
    important_words = ['agent', 'chop', 'deliver', 'coordinate', 'fast', 'slow', 'better', 'worse', 
                      'green', 'rose', 'blue', 'tomato', 'lettuce', 'onion', 'plate', 'knife']
    
    weighted_overlap = 0.0
    total_weight = 0.0
    
    for word in set1 & set2:
        weight = 2.0 if word in important_words else 1.0
        weighted_overlap += weight
        total_weight += weight
    
    for word in set1 | set2:
        if word not in (set1 & set2):
            weight = 2.0 if word in important_words else 1.0
            total_weight += weight
    
    semantic_sim = weighted_overlap / total_weight if total_weight > 0 else 0.0
    
    # 5. Try OpenAI embedding-based similarity if API key available
    embedding_sim = 0.0
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # Get embeddings for both sentences
            response1 = client.embeddings.create(
                input=sentence1,
                model="text-embedding-3-small"
            )
            response2 = client.embeddings.create(
                input=sentence2,
                model="text-embedding-3-small"
            )
            
            emb1 = np.array(response1.data[0].embedding)
            emb2 = np.array(response2.data[0].embedding)
            
            # Calculate cosine similarity
            embedding_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            embedding_sim = max(0.0, embedding_sim)  # Ensure non-negative
            
    except Exception as e:
        print(f"Warning: Could not compute embedding similarity: {e}")
        embedding_sim = 0.0
    
    # Combine similarities with weights
    if embedding_sim > 0:
        # Use embedding similarity as primary with lexical features as support
        final_similarity = (0.6 * embedding_sim + 0.15 * jaccard_sim + 
                          0.15 * cosine_sim + 0.1 * semantic_sim)
    else:
        # Fallback to lexical similarities
        final_similarity = (0.3 * jaccard_sim + 0.3 * cosine_sim + 
                          0.2 * edit_sim + 0.2 * semantic_sim)
    
    return min(1.0, max(0.0, final_similarity))
    
    
