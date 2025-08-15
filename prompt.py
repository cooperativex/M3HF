# Task Description Prompt
# Task Description Prompt
TASK_DESCRIPTION_A = """
You are an AI assistant helping to manage an Overcooked environment with multiple agents. 

The task is to prepare and deliver a {task_name}. 

The environment is a 7x7 grid with various objects and {num_agents} agents.

Observation Space (32-length vector for each agent):
- Tomato: position (2), status (1)
- Lettuce: position (2), status (1)
- Onion: position (2), status (1)
- Plate 1: position (2)
- Plate 2: position (2)
- Knife 1: position (2)
- Knife 2: position (2)
- Delivery: position (2)
- Agent 1: position (2)
- Agent 2: position (2)
- Agent 3: position (2)
- Order: one-hot encoded (7)

MA-V1 Actions (index indicates macro action):

0. Stay: 
   - The agent remains in its current position.
   - This is one of the five one-step macro-actions that are the same as primitive actions.

1. Get tomato:
   - Navigates the agent to the latest observed position of the tomato.
   - Picks the tomato up if it is there.
   - If not found at the latest observed position, the agent moves to check the initial position of the tomato.
   - Termination conditions:
     - The agent successfully picks up a chopped or unchopped tomato.
     - The agent observes the tomato is held by another agent or itself.
     - The agent is holding something else in hand.
     - The agent's path to the tomato is blocked by another agent.
     - The agent does not find the tomato either at the latest observed location or the initial location.
     - The agent attempts to enter the same cell with another agent but has a lower priority.

2. Get lettuce:
   - Identical behavior to "Get tomato", but for lettuce.
   - Same termination conditions as "Get tomato".

3. Get onion:
   - Identical behavior to "Get tomato", but for onion.
   - Same termination conditions as "Get tomato".

4. Get plate 1:
   - Navigates the agent to the latest observed position of plate 1.
   - Picks the plate up if it is there.
   - If not found at the latest observed position, the agent moves to check the initial position of the plate.
   - Termination conditions:
     - The agent successfully picks up a plate.
     - The agent observes the target plate is held by another agent or itself.
     - The agent is holding something else in hand.
     - The agent's path to the plate is blocked by another agent.
     - The agent does not find the plate either at the latest observed location or at the initial location.
     - The agent attempts to enter the same cell with another agent but has a lower priority.

5. Get plate 2:
   - Identical behavior to "Get plate 1", but for plate 2.
   - Same termination conditions as "Get plate 1".

6. Go to knife 1 (Go-Cut-Board-1):
   - Navigates the agent to the corresponding cutting board (knife 1).
   - Termination conditions:
     - The agent stops in front of the corresponding cutting board and places an in-hand item on it if the cutting board is not occupied.
     - If any other agent is using the target cutting board, the agent stops next to the teammate.
     - The agent attempts to enter the same cell with another agent but has a lower priority.

7. Go to knife 2 (Go-Cut-Board-2):
   - Identical behavior to "Go to knife 1", but for knife 2 (cutting board 2).
   - Same termination conditions as "Go to knife 1".

8. Deliver:
   - Navigates the agent to the 'star' cell for delivering.
   - Termination conditions:
     - The agent places the in-hand item on the cell if it is holding any item.
     - If any other agent is standing in front of the 'star' cell, the agent stops next to the teammate.
     - The agent attempts to enter the same cell with another agent but has a lower priority.

9. Chop:
   - Cuts a raw vegetable into pieces (taking three time steps) when the agent stands next to a cutting board and an unchopped vegetable is on the board.
   - If conditions are not met, it does nothing.
   - Termination conditions:
     - The vegetable on the cutting board has been chopped into pieces.
     - The agent is not next to a cutting board.
     - There is no unchopped vegetable on the cutting board.
     - The agent holds something in hand.

10. Right:
    - Moves the agent one cell to the right.
    - This is one of the five one-step macro-actions that are the same as primitive actions.

11. Down:
    - Moves the agent one cell down.
    - This is one of the five one-step macro-actions that are the same as primitive actions.

12. Left:
    - Moves the agent one cell to the left.
    - This is one of the five one-step macro-actions that are the same as primitive actions.

13. Up:
    - Moves the agent one cell up.
    - This is one of the five one-step macro-actions that are the same as primitive actions.

These macro-actions allow for more complex behaviors and interactions between agents in the Overcooked environment. They combine navigation, object interaction, and task completion into higher-level actions, enabling more efficient and strategic gameplay.
"""



# Task Description Prompt
TASK_DESCRIPTION_BC = """
You are an AI assistant helping to manage an Overcooked environment with multiple agents. 

The task is to prepare and deliver a {task_name}. 

The environment is a 7x7 grid with various objects and {num_agents} agents.

Observation Space (32-length vector for each agent):
- Tomato: position (2), status (1)
- Lettuce: position (2), status (1)
- Onion: position (2), status (1)
- Plate 1: position (2)
- Plate 2: position (2)
- Knife 1: position (2)
- Knife 2: position (2)
- Delivery: position (2)
- Agent 1: position (2)
- Agent 2: position (2)
- Agent 3: position (2)
- Order: one-hot encoded (7)

MA-V1 Actions (index indicates macro action):

0. Stay: 
   - The agent remains in its current position.
   - This is one of the five one-step macro-actions that are the same as primitive actions.

1. Get tomato:
   - Navigates the agent to the latest observed position of the tomato.
   - Picks the tomato up if it is there.
   - If not found at the latest observed position, the agent moves to check the initial position of the tomato.
   - Termination conditions:
     - The agent successfully picks up a chopped or unchopped tomato.
     - The agent observes the tomato is held by another agent or itself.
     - The agent is holding something else in hand.
     - The agent's path to the tomato is blocked by another agent.
     - The agent does not find the tomato either at the latest observed location or the initial location.
     - The agent attempts to enter the same cell with another agent but has a lower priority.

2. Get lettuce:
   - Identical behavior to "Get tomato", but for lettuce.
   - Same termination conditions as "Get tomato".

3. Get onion:
   - Identical behavior to "Get tomato", but for onion.
   - Same termination conditions as "Get tomato".

4. Get plate 1:
   - Navigates the agent to the latest observed position of plate 1.
   - Picks the plate up if it is there.
   - If not found at the latest observed position, the agent moves to check the initial position of the plate.
   - Termination conditions:
     - The agent successfully picks up a plate.
     - The agent observes the target plate is held by another agent or itself.
     - The agent is holding something else in hand.
     - The agent's path to the plate is blocked by another agent.
     - The agent does not find the plate either at the latest observed location or at the initial location.
     - The agent attempts to enter the same cell with another agent but has a lower priority.

5. Get plate 2:
   - Identical behavior to "Get plate 1", but for plate 2.
   - Same termination conditions as "Get plate 1".

6. Go to knife 1 (Go-Cut-Board-1):
   - Navigates the agent to the corresponding cutting board (knife 1).
   - Termination conditions:
     - The agent stops in front of the corresponding cutting board and places an in-hand item on it if the cutting board is not occupied.
     - If any other agent is using the target cutting board, the agent stops next to the teammate.
     - The agent attempts to enter the same cell with another agent but has a lower priority.

7. Go to knife 2 (Go-Cut-Board-2):
   - Identical behavior to "Go to knife 1", but for knife 2 (cutting board 2).
   - Same termination conditions as "Go to knife 1".

8. Deliver:
   - Navigates the agent to the 'star' cell for delivering.
   - Termination conditions:
     - The agent places the in-hand item on the cell if it is holding any item.
     - If any other agent is standing in front of the 'star' cell, the agent stops next to the teammate.
     - The agent attempts to enter the same cell with another agent but has a lower priority.

9. Chop:
   - Cuts a raw vegetable into pieces (taking three time steps) when the agent stands next to a cutting board and an unchopped vegetable is on the board.
   - If conditions are not met, it does nothing.
   - Termination conditions:
     - The vegetable on the cutting board has been chopped into pieces.
     - The agent is not next to a cutting board.
     - There is no unchopped vegetable on the cutting board.
     - The agent holds something in hand.

10. Go to counter:
    - Navigates the agent to the center cell in the middle of the map when the cell is not occupied.
    - If occupied, it moves to an adjacent cell.
    - If the agent is holding an object, the object will be placed on the counter.
    - If an object is on the counter, the object will be picked up.
    - This action is specific to the BC version of the environment.

11. Right:
    - Moves the agent one cell to the right.
    - This is one of the five one-step macro-actions that are the same as primitive actions.

12. Down:
    - Moves the agent one cell down.
    - This is one of the five one-step macro-actions that are the same as primitive actions.

13. Left:
    - Moves the agent one cell to the left.
    - This is one of the five one-step macro-actions that are the same as primitive actions.

14. Up:
    - Moves the agent one cell up.
    - This is one of the five one-step macro-actions that are the same as primitive actions.

These macro-actions allow for more complex behaviors and interactions between agents in the Overcooked environment. They combine navigation, object interaction, and task completion into higher-level actions, enabling more efficient and strategic gameplay. The main difference between this BC version and the A version is the inclusion of the "Go to counter" action, which provides an additional interaction point in the middle of the map.
"""

REWARD_FUNCTION_BUILD_PROMPT_v2 = """

REWARD_FUNCTION_BUILD_PROMPT
Given the parsed feedback for an agent in an Overcooked environment, create a conditional reward function. The observation space is a 32-length vector as described below.
Parsed Feedback: {feedback}
Observation Space (32-length vector for each agent):
Tomato: position (2), status (1) (obs[0:3])
Lettuce: position (2), status (1) (obs[3:6])
Onion: position (2), status (1) (obs[6:9])
Plate 1: position (2) (obs[9:11])
Plate 2: position (2) (obs[11:13])
Knife 1: position (2) (obs[13:15])
Knife 2: position (2) (obs[15:17])
Delivery: position (2) (obs[17:19])
Agent 1: position (2) (obs[19:21])
Agent 2: position (2) (obs[21:23])
Agent 3: position (2) (obs[23:25])
Order: one-hot encoded (7) (obs[25:32])
Actions are represented by integers from 0 to 14 as described in the previous prompts.
Available function templates with examples:
Conditional Distance-based:
   (reward if condition else 0)
   Example: (-sqrt((obs[19] - obs[0])**2 + (obs[20] - obs[1])**2) if obs[19] != obs[0] or obs[20] != obs[1] else 0)  # Distance reward to tomato only if not holding it
Conditional Action-based:
   (reward if action_condition and state_condition else 0)
   Example: (1 if act == 1 and obs[19] != obs[0] and obs[20] != obs[1] else 0)  # Reward for 'Get tomato' action only if not already holding it
Conditional Status-based:
   (reward if status_condition and state_condition else 0)
   Example: (1 if obs[2] == 1 and obs[19] == obs[0] and obs[20] == obs[1] else 0)  # Reward if tomato is chopped and agent is holding it
Conditional Proximity-based:
   (reward if proximity_condition and state_condition else 0)
   Example: (0.5 if sqrt((obs[19] - obs[13])**2 + (obs[20] - obs[14])**2) <= 1 and obs[19] == obs[0] and obs[20] == obs[1] else 0)  # Reward if agent 1 is near knife 1 while holding tomato
Conditional Holding-based (inferred):
   (reward if holding_condition else 0)
   Example: (1 if obs[19] == obs[0] and obs[20] == obs[1] else 0)  # Reward if agent 1 is holding tomato
Composite conditional reward:
   lambda obs, act: (
       (term1 if condition1 else 0) +
       (term2 if condition2 else 0) +
       ...
       (termN if conditionN else 0)
   )
Create a reward function by combining these templates based on the feedback. Return your response as a Python lambda function that takes the observation vector (obs) and action (act).
Ensure that your function uses the correct indices from the observation vector as described in the observation space.
Provide a detailed explanation of your choice and parameterization, including:
Why you chose specific templates and conditions
How the reward function handles different aspects and stages of the task
How the conditions ensure the correct order of actions
Any assumptions made about the task or environment
Use only basic mathematical operations and functions (e.g., +, -, *, /, max, min, sqrt).
Example format:
lambda obs, act: (
    (term1 if condition1 else 0) +
    (term2 if condition2 else 0) +
    ...
    (termN if conditionN else 0)
)
Where each term is a complete expression that contributes to the reward, and each condition ensures the reward is given only when appropriate.
Note: Ensure that all conditions and rewards are based solely on the observation vector and action. Do not use any external state information that is not provided in the observation vector.

Note: make sure to include the condition for each term in the reward function. And make sure the reward function is valid.
"""


FEEDBACK_PARSING_PROMPT = """
Given the following feedback for a multi-agent system in an Overcooked environment, 
assign the human feedback to appropriate agents or to all agents. The system has {num_agents} agents.

Human Feedback: {feedback}

Agent colors:
- agent_0 is Green0
- agent_1 is Rose or Pink or Red
- agent_2 is Blue

Guidelines for parsing:
1. Assign feedback to specific agents when they are explicitly mentioned by color or number.
2. Use "all" for general feedback that applies to all agents.
3. If a color is mentioned without a number, use the corresponding agent number (e.g., "Green" refers to agent_0).
4. Only include keys for agents that receive specific feedback, and 'all' if there's general feedback.
5. If there is no feedback for and agent, do not include the key for that agent. If there is no feedback for all agents, do not include the key for 'all'.

Return your response in the following JSON format:
{{
    "agent_0": "feedback for agent 0 (Green)",
    "agent_1": "feedback for agent 1 (Rose)",
    "agent_2": "feedback for agent 2 (Blue)",
    "all": "feedback for all agents"
}}


"""

REWARD_FUNCTION_BUILD_PROMPT_v1 = """

# REWARD_FUNCTION_BUILD_PROMPT

Given the parsed feedback for an agent in an Overcooked environment, select and parameterize
a reward function template. The observation space is a 32-length vector as described below.

Parsed Feedback: {feedback}

Observation Space (32-length vector for each agent):
- Tomato: position (2), status (1) (obs[0:3])
- Lettuce: position (2), status (1) (obs[3:6])
- Onion: position (2), status (1) (obs[6:9])
- Plate 1: position (2) (obs[9:11])
- Plate 2: position (2) (obs[11:13])
- Knife 1: position (2) (obs[13:15])
- Knife 2: position (2) (obs[15:17])
- Delivery: position (2) (obs[17:19])
- Agent 1: position (2) (obs[19:21])
- Agent 2: position (2) (obs[21:23])
- Agent 3: position (2) (obs[23:25])
- Order: one-hot encoded (7) (obs[25:32])

Actions are represented by integers from 0 to 14 as described in the previous prompts.

Available function templates with examples:

1. Distance-based:
   lambda obs, act: -sqrt((obs[e1_x] - obs[e2_x])**2 + (obs[e1_y] - obs[e2_y])**2)
   Example: lambda obs, act: -sqrt((obs[19] - obs[0])**2 + (obs[20] - obs[1])**2)  # Distance between agent 1 and tomato

2. Action-based:
   lambda obs, act: 1 if act == desired_action else 0
   Example: lambda obs, act: 1 if act == 1 else 0  # Reward for 'Get tomato' action

3. Status-based:
   lambda obs, act: 1 if obs[e_status] == desired_status else 0
   Example: lambda obs, act: 1 if obs[2] == 1 else 0  # Reward if tomato is chopped

4. Proximity-based:
   lambda obs, act: r_prox if sqrt((obs[e1_x] - obs[e2_x])**2 + (obs[e1_y] - obs[e2_y])**2) <= d else 0
   Example: lambda obs, act: 0.5 if sqrt((obs[19] - obs[13])**2 + (obs[20] - obs[14])**2) <= 1 else 0  # Reward if agent 1 is near knife 1

5. Holding-based (inferred):
   lambda obs, act: 1 if obs[agent_x] == obs[item_x] and obs[agent_y] == obs[item_y] else 0
   Example: lambda obs, act: 1 if obs[19] == obs[0] and obs[20] == obs[1] else 0  # Reward if agent 1 is holding tomato

6. Composite reward:
   lambda obs, act: sum(weight_i * reward_function_i(obs, act) for i in range(n))
   Example: lambda obs, act: (
       -0.5 * sqrt((obs[19] - obs[0])**2 + (obs[20] - obs[1])**2) +  # Distance to tomato
       1 if act == 1 else 0 +  # Reward for 'Get tomato'
       0.5 if obs[2] == 1 else 0  # Reward if tomato is chopped
   )

Select a template or combine multiple templates and parameterize them based on the feedback. 
Return your response as a Python lambda function that takes the observation vector (obs) and action (act).

Ensure that your function uses the correct indices from the observation vector as described in the observation space.
Provide a detailed explanation of your choice and parameterization, including:
1. Why you chose specific templates
2. How the reward function handles different aspects of the task
3. Any assumptions made about the task or environment

Use only basic mathematical operations and functions (e.g., +, -, *, /, max, min, sqrt).

Example format:
lambda obs, act: (
    (term1) +
    (term2) +
    ...
    (termN)
)

Where each term is a complete expression that contributes to the reward.

Note: Ensure that all conditions and rewards are based solely on the observation vector and action. Do not use any external state information that is not provided in the observation vector.

"""



REWARD_FUNCTION_BUILD_PROMPT = """
Given the parsed feedback for an agent in an Overcooked environment, select and parameterize
a reward function template. The observation space is a 32-length vector as described below.

Parsed Feedback: {feedback}

Observation Space (32-length vector for each agent):
- Tomato: position (2), status (1) (obs[0:3])
- Lettuce: position (2), status (1) (obs[3:6])
- Onion: position (2), status (1) (obs[6:9])
- Plate 1: position (2) (obs[9:11])
- Plate 2: position (2) (obs[11:13])
- Knife 1: position (2) (obs[13:15])
- Knife 2: position (2) (obs[15:17])
- Delivery: position (2) (obs[17:19])
- Agent 1: position (2) (obs[19:21])
- Agent 2: position (2) (obs[21:23])
- Agent 3: position (2) (obs[23:25])
- Order: one-hot encoded (7) (obs[25:32])

Available function templates with examples:

1. Distance-based:
   lambda obs, act: -sqrt((obs[e1_x] - obs[e2_x])**2 + (obs[e1_y] - obs[e2_y])**2)
   Example: lambda obs, act: -sqrt((obs[19] - obs[0])**2 + (obs[20] - obs[1])**2)  # Distance between agent 1 and tomato

2. Action-based:
   lambda obs, act: 1 if act == desired_action else 0
   Example: lambda obs, act: 1 if act == 5 else 0  # Reward for 'Interact' action

3. Status-based:
   lambda obs, act: 1 if obs[e_status] == desired_status else 0
   Example: lambda obs, act: 1 if obs[2] == 1 else 0  # Reward if tomato is chopped

4. Proximity-based:
   lambda obs, act: r_prox if sqrt((obs[e1_x] - obs[e2_x])**2 + (obs[e1_y] - obs[e2_y])**2) <= d else 0
   Example: lambda obs, act: 0.5 if sqrt((obs[19] - obs[13])**2 + (obs[20] - obs[14])**2) <= 1 else 0  # Reward if agent 1 is near knife 1

5. Time-based penalty:
   lambda obs, act, t: -beta * t
   Example: lambda obs, act, t: -0.001 * t  # Increasing penalty over time

6. Success-based:
   lambda obs, act: r_success if goal_condition_met(obs) else 0
   Example: lambda obs, act: 10 if obs[25] == 1 and obs[17] == obs[19] and obs[18] == obs[20] else 0  # Reward for delivering completed order

7. Energy-based penalty:
   lambda obs, act: -gamma * energy_cost(act)
   Example: lambda obs, act: -0.2 * (1 if act != 0 else 0)  # Penalty for non-zero actions

8. Composite reward:
   lambda obs, act: sum(weight_i * reward_function_i(obs, act) for i in range(n))
   Example: lambda obs, act: (
       -0.5 * sqrt((obs[19] - obs[0])**2 + (obs[20] - obs[1])**2) +  # Distance to tomato
       1 if act == 5 else 0 +  # Reward for 'Interact'
       0.5 if obs[2] == 1 else 0  # Reward if tomato is chopped
   )




Select a template or combine multiple templates and parameterize them based on the feedback. 
Return your response as a Python lambda function that takes the observation vector (obs), 
action (act), and if necessary, additional parameters like time step (t).

Ensure that your function uses the correct indices from the observation vector as described in the observation space.
Provide a brief explanation of your choice and parameterization.

Use only basic mathematical operations and functions (e.g., +, -, *, /, max, min, sqrt).

Example format:
lambda obs, act, t: (term1) + (term2) + ... + (termN)

Where each term is a complete expression that contributes to the reward.

"""


# Function to generate prompts
def generate_prompts(num_agents, feedback):
    prompts = {
        "feedback_parsing": FEEDBACK_PARSING_PROMPT.format(num_agents=num_agents, feedback=feedback),
        "reward_function_build": REWARD_FUNCTION_BUILD_PROMPT_v1 # REWARD_FUNCTION_BUILD_PROMPT_v2 #REWARD_FUNCTION_BUILD_PROMPT
    }
    return prompts


