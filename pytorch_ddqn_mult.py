#!/bin/env python
#import gymnasium as gym
import gym
import gym_sf
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import platform
import sys

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


###########################
# Replay Memory
###########################
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        # Save a transition
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

###########################
# Neural Network
###########################
class Network(nn.Module):

    def __init__(self, n_observations, n_actions, hid_dim):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(n_observations, hid_dim)
        self.layer2 = nn.Linear(hid_dim, hid_dim)
        self.layer3 = nn.Linear(hid_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


###########################
# Agent
###########################
class Agent:

    def __init__(self, env, batch_size, gamma, eps_start, eps_end, eps_decay, tau, 
                 lr, hid_dim=128, capacity=10_000, alg=['ddqn'], log_dir='logs/'):
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau

        self.alg = alg

        # Take the square of num_actions, since now the action space is individualized
        self.n_actions = int(env.action_space.n ** 0.5)
        state, _ = env.reset()
        n_observations = len(state)  # Why don't get env.observation_space.box (Generalize to discrete and continuous environmen)

        # Networks for the extended agents 1 and 2
        # Network has extended input n_observations + n_actions (Consider a1 and a2 have the same dimensions)
        self.policy_net1 = Network(n_observations, self.n_actions, hid_dim)
        self.target_net1 = Network(n_observations, self.n_actions, hid_dim)
        self.target_net1.load_state_dict(self.policy_net1.state_dict())

        self.policy_net2 = Network(n_observations + 1, self.n_actions, hid_dim)
        self.target_net2 = Network(n_observations + 1, self.n_actions, hid_dim)
        self.target_net2.load_state_dict(self.policy_net2.state_dict())

        self.optimizer1 = optim.AdamW(self.policy_net1.parameters(), lr=lr, amsgrad=True)
        self.optimizer2 = optim.AdamW(self.policy_net2.parameters(), lr=lr, amsgrad=True)

        self.memory = ReplayMemory(capacity=capacity)

        # If gpu is to be used
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.steps_done = 0

        self.episode_durations = []

        # Variables to collect statistics
        self.writer = SummaryWriter(log_dir)
        self.loss1 = torch.zeros(1)
        self.loss2 = torch.zeros(1)


    def select_action(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                # t.max(1) --> Largest column value of each row.
                # [1] Second column on max --> Index where max element was found, thus we pick the action that maximizes reward
                # view(1, 1) --> Reshape a tensor to have torch.Size([1, 1]) instead of torch.Size([])
                action1 = self.policy_net1(state).max(1)[1].view(1, 1)
                state_action1 = torch.cat((state, action1), dim=1)
                action2 = self.policy_net2(state_action1).max(1)[1].view(1, 1)
                action = action1 + self.n_actions * action2

                return action
        else:

            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    
    def plot_durations(self, show_result=False, save=False):
        plt.figure(1)
        # I believe that durations_t here acts is the reward, since the example will be with the cartpole environment
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.clf()
            plt.title('Result')
        else:
            # plt.clf() clears the current figure 
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episodes averages and plot them too
        if len(durations_t) >= 100:
            # .unfold(0, 100, 1) --> Sliding window view of size 100 of original tensor. Step size is 1 along first dimension
            # .mean(1) --> Compute the mean of each window along the second dimension
            # .view(-1) --> Reshapes the tensor to have a single dimension, flattening the tensor into a 1D array
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            # means.numpy() --> Returns a numpy array representation of the tensor means
            plt.plot(means.numpy())

        plt.pause(0.001) # pause a bit so that plots are updated

        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

        if save:
            unique_id = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
            plt.savefig(os.path.join(os.getcwd(), 'tutorial_pytorch', 'figures', unique_id + '__reward.png'))
            # self.writer.add_image('reward_plot', plt.figure(1), i_episode)
            

    def extend_state(self, state_batch, action_batch):
        # Extract first and second values from dictionary
        action1_batch = torch.tensor([self.env.action_dict[str(key.item())][0] for key in action_batch.flatten()])       
        action2_batch = torch.tensor([self.env.action_dict[str(key.item())][1] for key in action_batch.flatten()])

        action1_batch = action1_batch.reshape(action_batch.shape)
        action2_batch = action2_batch.reshape(action_batch.shape)

        state_action1_batch = torch.cat((state_batch, action1_batch), dim=1)

        return state_action1_batch, action1_batch, action2_batch

    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Converts batch-array of Transitions to Transitons of batch-arrays
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Individualize the actions and augment the state
        state_action1_batch, action1_batch, action2_batch = self.extend_state(state_batch, action_batch)
        # state_batch
        # state_action1_batch
        # action1_batch
        # action2_batch


        # --- Core of the DQN algorithm -------
        state_action1_values = self.policy_net1(state_batch).gather(1, action1_batch)
        state_action1_action2_values = self.policy_net2(state_action1_batch).gather(1, action2_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)

        if 'dqn' in self.alg:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net1(non_final_next_states).max(1)[0]
                expected_state_action1_values = self.target_net2(state_action1_batch).max(1)[0]

        if 'ddqn' in self.alg:
            # next_action_values = torch.zeros(self.batch_size, device=self.device)
            with torch.no_grad():
                next_action_values = self.policy_net(non_final_next_states).argmax(1).view(-1, 1)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_action_values).view(-1)

        # Very bad performance. It is probably wrong
        if 'ddqn2' in self.alg:
            # next_action_values = torch.zeros(self.batch_size, device=self.device)
            with torch.no_grad():
                next_action_values = self.target_net(non_final_next_states).argmax(1).view(-1, 1)
                next_state_values[non_final_mask] = self.policy_net(non_final_next_states).gather(1, next_action_values).view(-1)

        expected_state_action1_action2_values = (next_state_values * self.gamma) + reward_batch
        # --- Core of the DQN algorithm -------



        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        self.loss1 = criterion(state_action1_values, expected_state_action1_values.unsqueeze(1))
        self.loss2 = criterion(state_action1_action2_values, expected_state_action1_action2_values.unsqueeze(1))  

        # Optimize the model
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        self.loss1.backward()
        self.loss2.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net1.parameters(), 100)
        torch.nn.utils.clip_grad_value_(self.policy_net2.parameters(), 100)
        self.optimizer1.step()
        self.optimizer2.step()
        

    def train(self, num_episodes):
        if torch.cuda.is_available():
            num_episodes *= 10

        for i_episode in range(num_episodes):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            loss_mean1 = 0
            loss_mean2 = 0
            cum_reward = 0
            cum_discounted_reward = 0
            cum_gamma = self.gamma
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                cum_reward += reward
                cum_discounted_reward += cum_gamma * reward
                cum_gamma *= self.gamma
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                loss_mean1 += (self.loss1.item() - loss_mean1) / (t + 1)
                loss_mean2 += (self.loss2.item() - loss_mean2) / (t + 1)


                # Soft update of the target network's weights ( Why not use torch.nn.utisl.soft_update() in the parameters?)
                target_net1_state_dict = self.target_net1.state_dict()
                policy_net1_state_dict = self.policy_net1.state_dict()
                for key in policy_net1_state_dict:
                    target_net1_state_dict[key] = policy_net1_state_dict[key] * self.tau + target_net1_state_dict[key] * (1 - self.tau)
                self.target_net1.load_state_dict(target_net1_state_dict)

                target_net2_state_dict = self.target_net2.state_dict()
                policy_net2_state_dict = self.policy_net2.state_dict()
                for key in policy_net2_state_dict:
                    target_net2_state_dict[key] = policy_net2_state_dict[key] * self.tau + target_net2_state_dict[key] * (1 - self.tau)
                self.target_net2.load_state_dict(target_net2_state_dict)
                ############################### Maybe write as method to avoid repetion

                if done:
                    self.episode_durations.append(t + 1)
                    # self.plot_durations()
                    self.writer.add_scalar('loss1', loss_mean1, i_episode) 
                    self.writer.add_scalar('loss2', loss_mean2, i_episode) 
                    self.writer.add_scalar('episode_len', t+1, i_episode)
                    self.writer.add_scalar('reward', cum_reward, i_episode)
                    self.writer.add_scalar('disc_reward', cum_discounted_reward, i_episode)
                    break
        self.writer.close()

        # print('Complete')
        # self.plot_durations(show_result=True)
        # plt.ioff()
        # plt.show()


if __name__ == '__main__':
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

    if platform.system() == 'Linux':
        alg = [sys.argv[1]] #['dqn']
        num_episodes = int(sys.argv[2]) #50
        num_trial = int(sys.argv[3])
        env_name = sys.argv[4]
        num_agents = int(sys.argv[5]) # Choose 1 or 2
    else:
        alg = ['dqn']
        num_episodes = 50
        num_trial = 3
        env_name = 'four-room-multiagent-v0'
        num_agents = 2  

    for i in range(num_trial):
        # env_name = 'CartPole-v1'
        # env_name = 'LunarLander-v2'
        render_mode = "rgb_array"
        max_step_episode = 500
        random_initial_position = False
        env = gym.make(env_name,
                       render_mode=render_mode,
                       max_episode_steps=max_step_episode,
                       random_initial_position=random_initial_position,
                       max_num_agents=num_agents,
                       video=False)
        # env = gym.make(env_name)
        # print('Created')

        # Set up matplotlib (Basically, check if it is running in a jupyter notebook)
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display

        # This turns on iteractive mode. Allows matplotlib to add data to the plot without blocks the execution, as happen with plt.show()
        # plt.ion()

        # Hyperparameters
        BATCH_SIZE = 4 # 128
        GAMMA = 0.99
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 1000
        TAU = 0.005  # Tau is the update rate of the target network
        LR = 1e-4

        hid_dim = 128
        capacity = 10_000

        unique_id = datetime.now().strftime("%Y_%m_%d__%H_%M_%S__%f")[:-4]
        name = f'logs/{env_name}_nag{num_agents}_{alg[0]}_nt{num_trial:03}_run_{unique_id}'
        system_name = platform.system()
        if platform.system() == 'Linux':
            log_dir = f'/data/p285087/drl_alg/{name}'
        else:
            log_dir = name


        my_dqn = Agent(env, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, hid_dim=hid_dim, capacity=capacity, alg=alg, log_dir=log_dir)
        my_dqn.train(num_episodes)


