import gym
import gym_sf
import torch
import torch.nn as nn
from pytorch_ddqn import Agent, Network
from datetime import datetime

import os
import time


def run_episode(episode_model, env_test):
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
        with torch.no_grad():
            action = model(next_state).max(1)[1].view(1, 1)
        next_state, reward, terminated, truncated, _ = env_test.step(action.item())
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
num_agents = 1
hid_dim = 256
job = '28'

path = f'models/{job}/my_model.pth'
video_path = f'models/{job}'
initial_position=[(12,0), (12, 0)] if num_agents == 2 else [(12,0)]
given_initial_position=True

# name = "four-room-v0"
# env = gym.make(name, render_mode="rgb_array_list", max_episode_steps=100, video=True, random_initial_position=False)
env = gym.make(name, render_mode="rgb_array_list", max_episode_steps=100, video=True,
                random_initial_position=False, max_num_agents=num_agents, video_path=video_path,
                given_initial_position=given_initial_position)


# Initialize the model
n_actions = env.action_space.n
state, _ = env.reset()
n_observations = len(state)
model = Network(n_observations, n_actions, hid_dim)

# Make sure to set the map_location if you saved the model on a different device (e.g., GPU) than the one you're loading it on (e.g., CPU)
state_dict = torch.load(path, map_location=torch.device('cpu'))

# Load the state_dict into your model
model.load_state_dict(state_dict)


for _ in range(3):
    run_episode(model, env)
    time.sleep(1)