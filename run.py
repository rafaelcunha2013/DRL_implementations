#!/bin/env python
import gym
import gym_sf

from pytorch_ddqn import Agent, AgentOneAtTime
# from pytorch_ddqn_per import Agent

import platform
import sys
from datetime import datetime


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
    nn = ['linear'] 
    capacity = int(sys.argv[7])
    hid_dim = int(sys.argv[8])
    EPS_DECAY = int(sys.argv[9])
    max_step_episode = int(sys.argv[10])
    buffer = [sys.argv[11]]
    agent_name = [sys.argv[12]]
    print(agent_name)

else:
    alg = ['dqn']
    num_episodes = 10
    num_trial = 1
    env_name = 'four-room-multiagent-v0'
    num_agents = 2 
    folder = 'logs5'
    nn = ['linear'] 
    capacity = 10_000
    hid_dim = 256
    EPS_DECAY = 1000
    max_step_episode = 500
    buffer = ['simple'] # or ['simple'] or ['per']
    agent_name = "AgentAtTime" # 'Agent' # or "AgentAtTime"

for i in range(num_trial):
    # env_name = 'CartPole-v1'
    # env_name = 'LunarLander-v2'      
    render_mode = "rgb_array"
    # max_step_episode = 500
    random_initial_position = False
    initial_position=[(12,0), (12, 0)] if num_agents == 2 else [(12,0)]
    given_initial_position=True
    # if given_initial_position:
    #     if num_agents == 2:
    #         initial_position=[(12,0), (12, 0)]
    #     else:
    #         initial_position=[(12,0)]

    env = gym.make(env_name,
                    render_mode=render_mode,
                    max_episode_steps=max_step_episode,
                    random_initial_position=random_initial_position,
                    max_num_agents=num_agents,
                    video=False,
                    given_initial_position=given_initial_position,
                    initial_position=initial_position)


    # Hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1.0
    EPS_END = 0.1
    # EPS_DECAY = 1000
    TAU = 0.005  # Tau is the update rate of the target network
    LR = 0.00025 #1e-4

    # hid_dim = 32  # When using with CNN (Atari Human architecture)
    # capacity = 10_000 #1_000_000  
    
    unique_id = datetime.now().strftime("%Y_%m_%d__%H_%M_%S__%f")[:-4]
    name = f'{folder}_{num_agents}/{env_name}_{alg[0]}_{capacity}_run_{unique_id}'
    # log_dir = f'/data/p285087/DRL_labs/{name}' if system_name == 'Linux' else name
    log_dir = f'/home4/p285087/data/four_room/{name}' if system_name == 'Linux' else name

    agent_classes = {'Agent': Agent, "AgentAtTime": AgentOneAtTime}
    my_agent = agent_classes[agent_name]

    if 'per' in buffer:
        my_dqn = my_agent(env, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, hid_dim=hid_dim, capacity=capacity, alg=alg, log_dir=log_dir, nn=nn, buffer=buffer)
    else:
        my_dqn = my_agent(env, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, hid_dim=hid_dim, capacity=capacity, alg=alg, log_dir=log_dir, nn=nn)


    my_dqn.train(num_episodes)