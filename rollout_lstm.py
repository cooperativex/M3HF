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


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

register_env("multi_agent_overcooked", lambda config: MultiAgentOvercookedEnv(config))
print("Environment registered")


def render_video_from_checkpoint(checkpoint_path, env_config, video_path, num_episodes=1, max_steps_per_episode=1000):
    logger.info("Starting render_video_from_checkpoint function")
    
    
    temp_env = MultiAgentOvercookedEnv(env_config.copy())
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    
    
    try:
        # Initialize Ray in local mode
        ray.init(ignore_reinit_error=True, local_mode=True, logging_level=logging.ERROR)
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
                train_batch_size=65536,
                sgd_minibatch_size=2048,
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
        config.model["use_lstm"] = True
        config.model["lstm_use_prev_action"] = True
        config.model["lstm_use_prev_reward"] = False
        config.model["lstm_cell_size"] = 512
        logger.info("PPO configuration set up successfully")

        # Load the trained algorithm
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        algo = config.build()
        algo.restore(checkpoint_path)
        logger.info("Checkpoint loaded successfully")

        # Set up video writer
        logger.info(f"Setting up video writer to save video at {video_path}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, 10, (700, 700))
        
        
        actionName = ["stay", "get tomato", "get lettuce", "get onion", "get plate 1", "get plate 2", "go to knife 1", "go to knife 2", "deliver", "chop", "right", "down", "left", "up"]
        
        output_reward = 0
        discount = 1
        step = 0

        for episode in range(num_episodes):
            logger.info(f"Starting episode {episode + 1}/{num_episodes}")
            obs = env.reset()
            done = {"__all__": False}
            step = 0
            state = {
                f"agent_{i}": algo.get_policy(f"agent_{i}").get_initial_state()
                for i in range(env_config['n_agent'])
            }
            prev_actions = {
                f"agent_{i}": 0 
                for i in range(env_config['n_agent'])
            }
            prev_rewards = {
                f"agent_{i}": 0.0  # Initialize with a default reward value
                for i in range(env_config['n_agent'])
            }

            while not done["__all__"] and step < max_steps_per_episode:
                actions = {}
                # print ("step: ", step)
                obs_copy = copy.deepcopy(obs)  # Create a deep copy of obs
                for agent_id in range(env_config['n_agent']):
                    policy_id = f"agent_{agent_id}"
                    
                    # print("The current obs is: ", obs_copy)
                    agent_obs = obs_copy[0][agent_id]
                    agent_obs = agent_obs.astype(np.float32)
                    
                    # print(f"agent {agent_id} obs: ", agent_obs)
                    
                    action, state[policy_id], _ = algo.compute_single_action(
                        observation=agent_obs,
                        state=state[policy_id],
                        prev_action=prev_actions[policy_id],
                        prev_reward=prev_rewards[policy_id],
                        policy_id=policy_id,
                        explore=False
                    )
                    
                    print("The action is: ", action)
                    actions[agent_id] = action
                    
                    # Store the action index for the next iterations
                    prev_actions[policy_id] = action

                obs, rewards, done, info = env.step(actions)

                print("action: ", actions)
                output_reward += discount * rewards[0]
                discount *= 0.99
                print("------------------------------------------------------------------")
                print("step: ", step, "rewards: ", output_reward)
                print("#############################################")
                print("Blue Agent Action: ", actionName[info['cur_mac'][0]])
                print("Action Done: ", info['mac_done'][0])
                print("Blue Agent Observation")
                print("tomato pos: ", obs[0][0:2]*7)
                print("tomato status: ", obs[0][2])
                print("lettuce pos: ", obs[0][3:5]*7)
                print("lettuce status: ", obs[0][5])
                print("onion pos: ", obs[0][6:8]*7)
                print("onion status: ", obs[0][8])            
                print("plate-1 pos: ", obs[0][9:11]*7)
                print("plate-2 pos: ", obs[0][11:13]*7)
                print("knife-1 pos: ", obs[0][13:15]*7)
                print("knife-2 pos: ", obs[0][15:17]*7)
                print("delivery: ", obs[0][17:19]*7)
                print("agent-1: ", obs[0][19:21]*7)
                print("agent-2: ", obs[0][21:23]*7)
                print("agent-3: ", obs[0][23:25]*7)
                print("order: ", obs[0][25:])
                print("#############################################")
                print("#############################################")
                print("Pink Agent Action: ", actionName[info['cur_mac'][1]])
                print("Action Done: ", info['mac_done'][1])
                print("Pink Agent Observation")
                print("tomato pos: ", obs[1][0:2]*7)
                print("tomato status: ", obs[1][2])
                print("lettuce pos: ", obs[1][3:5]*7)
                print("lettuce status: ", obs[1][5])
                print("onion pos: ", obs[1][6:8]*7)
                print("onion status: ", obs[1][8])            
                print("plate-1 pos: ", obs[1][9:11]*7)
                print("plate-2 pos: ", obs[1][11:13]*7)
                print("knife-1 pos: ", obs[1][13:15]*7)
                print("knife-2 pos: ", obs[1][15:17]*7)
                print("delivery: ", obs[1][17:19]*7)
                print("agent-1: ", obs[1][19:21]*7)
                print("agent-2: ", obs[1][21:23]*7)
                print("agent-3: ", obs[1][23:25]*7)
                print("order: ", obs[1][25:])
                print("#############################################")
                print("#############################################")
                print("Green Agent Action: ", actionName[info['cur_mac'][2]])
                print("Action Done: ", info['mac_done'][2])
                print("Pink Agent Observation")
                print("tomato pos: ", obs[2][0:2]*7)
                print("tomato status: ", obs[2][2])
                print("lettuce pos: ", obs[2][3:5]*7)
                print("lettuce status: ", obs[2][5])
                print("onion pos: ", obs[2][6:8]*7)
                print("onion status: ", obs[2][8])            
                print("plate-1 pos: ", obs[2][9:11]*7)
                print("plate-2 pos: ", obs[2][11:13]*7)
                print("knife-1 pos: ", obs[2][13:15]*7)
                print("knife-2 pos: ", obs[2][15:17]*7)
                print("delivery: ", obs[2][17:19]*7)
                print("agent-1: ", obs[2][19:21]*7)
                print("agent-2: ", obs[2][21:23]*7)
                print("agent-3: ", obs[2][23:25]*7)
                print("order: ", obs[2][25:])
                print("#############################################")
                print("#############################################")
                print("#############################################")
                print("#############################################")
                print()
                print()
                print()


                # Update previous rewards
                for agent_id in range(env_config['n_agent']):
                    policy_id = f"agent_{agent_id}"
                    prev_rewards[policy_id] = rewards[agent_id]

                # Render the frame
                frame = env.render(mode='rgb_array')
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(frame)

                step += 1
                
                
            
            logger.info(f"Episode {episode + 1} completed in {step} steps")

        video.release()
        env.close()
        logger.info("Video saved and environment closed")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

    finally:
        ray.shutdown()
        logger.info("Ray shutdown completed")





if __name__ == "__main__":
    env_config = {
        'env_id': 'Overcooked-MA-v1',
        'grid_dim': [7, 7],
        'task': "lettuce-onion-tomato salad",
        'rewardList': {"subtask finished": 10, "correct delivery": 200, "wrong delivery": -5, "step penalty": -0.1},
        'map_type': "A",
        'n_agent': 3,
        'obs_radius': 2,
        'mode': "vector",
        'debug': False
    }

    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "checkpoint_20") 
    video_path = "rendered_video.mp4"

    render_video_from_checkpoint(checkpoint_path, env_config, video_path)
    print(f"Video saved to {video_path}")