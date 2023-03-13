import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

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

        n_actions = env.action_space.n
        state, _ = env.reset()
        n_observations = len(state)  # Why don't get env.observation_space.box (Generalize to discrete and continuous environmen)

        self.policy_net = Network(n_observations, n_actions, hid_dim)
        self.target_net = Network(n_observations, n_actions, hid_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(capacity=capacity)

        # If gpu is to be used
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.steps_done = 0

        self.episode_durations = []

        # Variables to collect statistics
        self.writer = SummaryWriter(log_dir)
        self.loss = torch.zeros(1)


    def select_action(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                # t.max(1) --> Largest column value of each row.
                # [1] Second column on max --> Index where max element was found, thus we pick the action that maximizes reward
                # view(1, 1) --> Reshape a tensor to have torch.Size([1, 1]) instead of torch.Size([])
                return self.policy_net(state).max(1)[1].view(1, 1)
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


        # --- Core of the DQN algorithm -------
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)

        if 'dqn' in self.alg:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        if 'ddqn' in self.alg:
            # next_action_values = torch.zeros(self.batch_size, device=self.device)
            with torch.no_grad():
                next_action_values = self.policy_net(non_final_next_states).argmax(1).view(-1, 1)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_action_values).view(-1)

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # --- Core of the DQN algorithm -------



        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        self.loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        self.loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        

    def train(self, num_episodes):
        if torch.cuda.is_available():
            num_episodes *= 10

        for i_episode in range(num_episodes):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            loss_mean = 0
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                loss_mean += (self.loss.item() - loss_mean) / (t + 1)


                # Soft update of the target network's weights ( Why not use torch.nn.utisl.soft_update() in the parameters?)
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    # self.plot_durations()
                    self.writer.add_scalar('loss', loss_mean, i_episode) 
                    self.writer.add_scalar('reward', t+1, i_episode)
                    break
        self.writer.close()

        # print('Complete')
        # self.plot_durations(show_result=True)
        # plt.ioff()
        # plt.show()


if __name__ == '__main__':
    for i in range(7):
        env_name = 'CartPole-v1'
        env = gym.make(env_name)

        # Set up matplotlib (Basically, check if it is running in a jupyter notebook)
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display

        # This turns on iteractive mode. Allows matplotlib to add data to the plot without blocks the execution, as happen with plt.show()
        # plt.ion()

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
        num_episodes = 1000

        log_dir = 'logs/dqn_run' + str(i)
        my_dqn = Agent(env, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, hid_dim=hid_dim, capacity=capacity, alg=['dqn'], log_dir=log_dir)
        my_dqn.train(num_episodes)

        print('Complete')
        my_dqn.plot_durations(show_result=True, save=True)
        # plt.ioff()
        # plt.show()
    


