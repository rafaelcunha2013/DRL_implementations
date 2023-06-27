import gym
import gym_sf
import torch
import torch.nn as nn
from pytorch_ddqn import Agent, Network
from datetime import datetime

import os
import time


def run_episode(agent1, agent2, env_test):
    terminated = False
    truncated = False
    next_state, _ = env_test.reset()
    step = 0
    gamma = 0.99
    cum_gamma = gamma
    cum_discounted_reward = 0


    while True:
        step += 1
        next_state = torch.tensor(next_state, dtype=torch.float32, device='cpu').unsqueeze(0)
        next_state1 = torch.cat((next_state[0, 0:2].unsqueeze(0), next_state[0, 4:].unsqueeze(0)), dim=1)
        next_state2 = next_state[0, 2:].unsqueeze(0)
        with torch.no_grad():
            action1 = agent1(next_state1).max(1)[1].view(1, 1).item()
            action2 = agent2(next_state2).max(1)[1].view(1, 1).item()
        action = action1 + 4 * action2
        next_state, reward, terminated, truncated, _ = env_test.step(action)
        # print(f'({step}, {reward})')
        cum_discounted_reward += cum_gamma * reward
        cum_gamma *= gamma



        if terminated or truncated:
            env_test.render()
            env_test.reset()
            print(cum_discounted_reward)
            break
    env_test.close()

name = "four-room-multiagent-v0"
num_agents = 2
hid_dim = 256
job = '43/43_ddqn_Agent_06_26__15_04_04__22'

path1 = f'models/{job}/agent1.pth'
path2 = f'models/{job}/agent2.pth'
video_path = f'models/{job}'
initial_position=[(12,0), (12, 0)] if num_agents == 2 else [(12,0)]
given_initial_position=True

# name = "four-room-v0"
# env = gym.make(name, render_mode="rgb_array_list", max_episode_steps=100, video=True, random_initial_position=False)
env = gym.make(name, render_mode="rgb_array_list", max_episode_steps=100, video=True,
                random_initial_position=False, max_num_agents=num_agents, video_path=video_path,
                given_initial_position=given_initial_position, initial_position=initial_position)


# Initialize the model
n_actions = 4 #env.action_space.n
state, _ = env.reset()
n_observations = 14 #len(state)

agent1 = Network(n_observations, n_actions, hid_dim)
agent2 = Network(n_observations, n_actions, hid_dim)

# Make sure to set the map_location if you saved the model on a different device (e.g., GPU) than the one you're loading it on (e.g., CPU)
state_dict1 = torch.load(path1, map_location=torch.device('cpu'))
state_dict2 = torch.load(path2, map_location=torch.device('cpu'))

# Load the state_dict into your model
agent1.load_state_dict(state_dict1)
agent2.load_state_dict(state_dict2)


for _ in range(3):
    run_episode(agent1, agent2, env)
    time.sleep(1)