import argparse
import gym

from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper
import random


TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]

import cv2
import numpy as np
import os

def play(env_id, grid_dim, task, map_type, n_agent, obs_radius, mode, debug):

    rewardList = {"subtask finished": 10, "correct delivery": 200, "wrong delivery": -5, "step penalty": -0.1}
    env_params = {'grid_dim': grid_dim,
                    'task': TASKLIST[task],
                    'rewardList': rewardList,
                    'map_type': map_type,
                    'n_agent': n_agent,
                    'obs_radius': obs_radius,
                    'mode': mode,
                    'debug': debug
                }
    env = gym.make(env_id, **env_params)

    if env_id == "Overcooked-v1":
        #               0        1       2        3     4   
        actionName = ["stay", "right", "down", "left", "up"]
    else:
        env = MacEnvWrapper(env)
        if map_type == "A":
            #               0          1              2             3            4               5              6                7             8          9      10       11      12     13
            actionName = ["stay", "get tomato", "get lettuce", "get onion", "get plate 1", "get plate 2", "go to knife 1", "go to knife 2", "deliver", "chop", "right", "down", "left", "up"]
        else:
            #               0          1              2             3            4               5              6                7             8          9           10          11       12      13     14
            actionName = ["stay", "get tomato", "get lettuce", "get onion", "get plate 1", "get plate 2", "go to knife 1", "go to knife 2", "deliver", "chop", "go to counter", "right", "down", "left", "up"]

    rewards = 0
    discount = 1
    step = 0

    # Video recording setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_dir = "videos"
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{env_id}_gameplay.mp4")
    fps = 5  # Adjust as needed
    video_writer = None

    obs = env.reset()     
    frame = env.render(mode='rgb_array')
    
    if video_writer is None:
        h, w = frame.shape[:2]
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    while(True):
        step += 1
        # a = input("input:").split(" ")
        # action = [int(a[0]), int(a[1]), int(a[2])]
        # random pick one 
        # action = [int(a[0]), int(a[1]), int(a[2])]
        print(actionName)
        
        action = [random.randint(0, len(actionName)-1 ) for _ in range(n_agent)]
        
        print("action: ", action)
        obs, reward, done, info = env.step(action)
        frame = env.render(mode='rgb_array')
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        rewards += discount * reward[0]
        discount *= 0.99
        print("------------------------------------------------------------------")
        print("step: ", step, "rewards: ", reward)
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
        if done:
            break
        if step > 200:
            break

    video_writer.release()
    print(f"Video saved as {video_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id',                 action='store',        type=str,             default='Overcooked-MA-v1',    help='Domain name')
    parser.add_argument('--n_agent',                action='store',        type=int,             default=3,                     help='Number of agents')
    parser.add_argument('--grid_dim',               action='store',        type=int,  nargs=2,   default=[7,7],                 help='Grid world size')
    parser.add_argument('--task',                   action='store',        type=int,             default=6,                     help='The receipt agent cooks')
    parser.add_argument('--map_type',               action='store',        type=str,             default="A",                   help='The type of map')
    parser.add_argument('--obs_radius',             action='store',        type=int,             default=2,                     help='The radius of the agents')
    parser.add_argument('--mode',                   action='store',        type=str,             default="vector",              help='The type of the observation(vector/image)')    
    parser.add_argument('--debug',                  action='store',        type=bool,            default=True,                  help='Whehter print the debug information and render')                
   
    params = vars(parser.parse_args())
    play(**params)



