import gym
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper
from play_rllib_ippo import MultiAgentOvercookedEnv

class CustomRewardOvercookedEnv(MultiAgentOvercookedEnv):
    def __init__(self, env_config, current_reward_functions):
        super().__init__(env_config)
        self.current_reward_functions = current_reward_functions
        self.original_rewards = None  # To store the original rewards
        # self.generation = generation
        # self.max_steps = 200 * (self.generation + 1)

    def step(self, action_dict):
        # Use the parent class to get the original observations, rewards, dones, etc.
        obs, rewards, dones, truncateds, infos = super().step(action_dict)

        # Because this is a team reward, all agents have the same original reward
        # Store the original reward
        ori_reward = None
        if rewards:
            # Get any agent's reward
            ori_reward = next(iter(rewards.values()))
            self.original_rewards = ori_reward

        # Modify the rewards using the current_reward_functions
        if self.current_reward_functions:
            for agent_id in rewards:
                obs_agent = obs[agent_id]
                act_agent = action_dict[agent_id]
                t = self.step_count  # Assume step_count is managed in the parent class

                # Apply the custom reward function
                custom_reward_func = self.current_reward_functions.get(f"agent_{agent_id}", None)
                if custom_reward_func:
                    addtional_rew = custom_reward_func(obs_agent, act_agent, ori_reward, t)
                    # Scale down additional_rew if its absolute value is greater than 200
                    if abs(addtional_rew) > 200:
                        n = 0
                        while abs(addtional_rew) > 200:
                            addtional_rew /= 10
                            n += 1
                        addtional_rew = round(addtional_rew, 2)  # Round to 2 decimal places
                    # Divide by number of agents to prevent reward multiplication in multi-agent setting
                    # Since RLLib sums all agent rewards, we need to divide to get correct team reward scale
                    n_agents = len(rewards)
                    rewards[agent_id] = (addtional_rew + self.original_rewards) / n_agents 
                else:
                    # If no custom function, keep the original reward
                    rewards[agent_id] = ori_reward
        else:
            # No custom reward functions, use original reward divided by number of agents
            n_agents = len(rewards)
            for agent_id in rewards:
                rewards[agent_id] = ori_reward / n_agents

        # Add the original reward to the info dictionary for logging
        if infos is None:
            infos = {}
        if not isinstance(infos, dict):
            infos = {}
        if '__common__' not in infos:
            infos['__common__'] = {}
        infos['__common__']['original_reward'] = self.original_rewards 

        return obs, rewards, dones, truncateds, infos

    def get_original_rewards(self):
        # Method to retrieve the original rewards for logging
        return self.original_rewards
