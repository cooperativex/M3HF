import argparse
import gym
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper

TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]

class RLlibOvercookedEnv(gym.Env):
    def __init__(self, env_config):
        self.env = gym.make("Overcooked-v1", **env_config)
        self.env = MacEnvWrapper(self.env)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

def create_env(env_config):
    return RLlibOvercookedEnv(env_config)

def train_ppo(env_config, num_workers=2, num_gpus=0, training_iterations=200):
    ray.init()

    register_env("overcooked_env", lambda config: RLlibOvercookedEnv(config))

    config = (
        PPOConfig()
        .environment("overcooked_env", env_config=env_config)
        .framework("torch")
        .rollouts(num_rollout_workers=num_workers)
        .resources(num_gpus=num_gpus)
        .training(
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=30,
            lr=5e-5,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
        )
    )

    stop = {
        "training_iteration": training_iterations,
    }

    results = tune.run(
        "PPO",
        config=config.to_dict(),
        stop=stop,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        verbose=1,
    )

    best_checkpoint = results.get_best_checkpoint(results.get_best_trial("episode_reward_mean"), "episode_reward_mean")
    print(f"Best checkpoint: {best_checkpoint}")
    ray.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', action='store', type=str, default='Overcooked-MA-v1', help='Domain name')
    parser.add_argument('--n_agent', action='store', type=int, default=3, help='Number of agents')
    parser.add_argument('--grid_dim', action='store', type=int, nargs=2, default=[7,7], help='Grid world size')
    parser.add_argument('--task', action='store', type=int, default=6, help='The receipt agent cooks')
    parser.add_argument('--map_type', action='store', type=str, default="A", help='The type of map')
    parser.add_argument('--obs_radius', action='store', type=int, default=2, help='The radius of the agents')
    parser.add_argument('--mode', action='store', type=str, default="vector", help='The type of the observation(vector/image)')    
    parser.add_argument('--debug', action='store', type=bool, default=False, help='Whether print the debug information and render')
    parser.add_argument('--num_workers', action='store', type=int, default=2, help='Number of rollout workers')
    parser.add_argument('--num_gpus', action='store', type=int, default=3, help='Number of GPUs to use')
    parser.add_argument('--training_iterations', action='store', type=int, default=200, help='Number of training iterations')

    args = parser.parse_args()
    
    env_config = {
        'grid_dim': args.grid_dim,
        'task': TASKLIST[args.task],
        'rewardList': {"subtask finished": 10, "correct delivery": 200, "wrong delivery": -5, "step penalty": -0.1},
        'map_type': args.map_type,
        'n_agent': args.n_agent,
        'obs_radius': args.obs_radius,
        'mode': args.mode,
        'debug': args.debug
    }

    train_ppo(env_config, args.num_workers, args.num_gpus, args.training_iterations)