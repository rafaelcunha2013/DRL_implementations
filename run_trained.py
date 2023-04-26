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
    

    while True:
        next_state = torch.tensor(next_state, dtype=torch.float32, device='cpu').unsqueeze(0)
        with torch.no_grad():
            action = model(next_state).max(1)[1].view(1, 1)
        next_state, reward, terminated, truncated, _ = env_test.step(action.item())


        if terminated or truncated:
            env_test.render()
            env_test.reset()
            break
    env_test.close()

name = "four-room-multiagent-v0"
name = "four-room-v0"
env = gym.make(name, render_mode="rgb_array_list", max_episode_steps=100, video=True, random_initial_position=False)
# env = gym.make(name, render_mode="rgb_array_list", max_episode_steps=100, video=True,
#                random_initial_position=False, max_num_agents=1)


# Initialize the model
n_actions = env.action_space.n
state, _ = env.reset()
n_observations = len(state)
model = Network(n_observations, n_actions, 256)

# Make sure to set the map_location if you saved the model on a different device (e.g., GPU) than the one you're loading it on (e.g., CPU)
state_dict = torch.load("my_model.pth", map_location=torch.device('cpu'))

# Load the state_dict into your model
model.load_state_dict(state_dict)


for _ in range(3):
    run_episode(model, env)
    time.sleep(1)