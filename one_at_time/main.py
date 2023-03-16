#!/bin/env python
import gym
import gym_sf

from datetime import datetime
import os
import platform
import sys

from agents import Agent, AgentOneAtTime

"""
Intructions to run the code via command line
sbatch run_four_room.sh ddqn 10_000 5 four-room-multiagent-v0 1 
Arguments:
1: Algorithm used. Choose among 'dqn' or 'ddqn'. (alg)
2: Number of episodes for each run (num_episodes)
3: Number of runs in the for loop (num_trial)
4: Name of the environment (env_name)
5: Number of agents (num_agents)
"""

system_name = platform.system()
if system_name == 'Linux':
    alg = [sys.argv[1]] #['dqn']
    num_episodes = int(sys.argv[2]) #50
    num_trial = int(sys.argv[3])
    env_name = sys.argv[4]
    num_agents = int(sys.argv[5]) # Choose 1 or 2
    folder = sys.argv[6]
    agent_type = sys.argv[7]
else:
    alg = ['dqn']
    num_episodes = 4_000
    num_trial = 1
    env_name = 'four-room-multiagent-v0'
    num_agents = 2  
    folder = 'logs4'
    agent_type = 'bertsekas'

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005  # Tau is the update rate of the target network
LR = 1e-4

hid_dim = 128
capacity = 10_000

# env_name = 'CartPole-v1'
# env_name = 'LunarLander-v2'
render_mode = "rgb_array"
max_step_episode = 500
random_initial_position = False

for i in range(num_trial):

    env = gym.make(env_name,
                    render_mode=render_mode,
                    max_episode_steps=max_step_episode,
                    random_initial_position=random_initial_position,
                    max_num_agents=num_agents,
                    video=False)
    # env = gym.make(env_name)
    # print('Created')
    #folder/
    unique_id = datetime.now().strftime("%Y_%m_%d__%H_%M_%S__%f")[:-4]
    name = f'{folder}/{env_name}_nag{num_agents}_{alg[0]}_{agent_type}_run_{unique_id}'
    log_dir = f'/data/p285087/drl_alg/{name}' if system_name == 'Linux' else name

    if agent_type == 'regular':
        my_dqn = Agent(env, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, hid_dim=hid_dim, capacity=capacity, alg=alg, log_dir=log_dir)
    else:
        my_dqn = AgentOneAtTime(env, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, hid_dim=hid_dim, capacity=capacity, alg=alg, log_dir=log_dir)
    my_dqn.train(num_episodes)
