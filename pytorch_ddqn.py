#!/bin/env python
# import gymnasium as gym
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

from replay_buffer import PrioritizeExperienceReplay


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

#####################################
# Dueling Neural Network
#####################################
class DuelNetwork(nn.Module):
    def __init__(self, n_observations, n_actions, hid_dim):
        super(DuelNetwork, self).__init__()

        self.feature_layer = nn.Linear(n_observations, hid_dim)

        # Action value function stream
        self.advantage_layer1 = nn.Linear(hid_dim, hid_dim)
        self.advantage_layer2 = nn.Linear(hid_dim, n_actions)

        # State value function stream
        self.value_layer1 = nn.Linear(hid_dim, hid_dim)
        self.value_layer2 = nn.Linear(hid_dim, 1)

    
    def forward(self, x):
        x = F.relu(self.feature_layer(x))

        advantage = F.relu(self.advantage_layer1(x))
        advantage = self.advantage_layer2(advantage)

        value = F.relu(self.value_layer1(x))
        value = self.value_layer2(value)

        # Recombine streams
        return value + advantage - advantage.mean()


###########################
# Convolutional Neural Network
###########################
class CNN(nn.Module):

    def __init__(self, n_channels, n_actions):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, n_actions)



    def forward(self, x):
        # x should have [batch_size, channels, height, width]
        # Change the order from [1, 84, 84, 4] to [1, 4, 84, 84]
        x = x.permute(0, 3, 1, 2)  
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.reshape(x.size(0), -1)))

        return self.fc5(x)

    

###########################
# Agent
###########################
class Agent:

    def __init__(self, env, batch_size, gamma, eps_start, eps_end, eps_decay, tau, 
                 lr, hid_dim=128, capacity=10_000, alg=['ddqn'], log_dir='logs/', nn=['CNN'], n_agents=1,
                 buffer=['simple']):
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.alg = alg
        self.log_dir = log_dir
        self.n_agents = n_agents
        self.buffer = buffer


        self.eps_threshold = None
        self.evaluate_flag = False       
        self.n_actions = int(self.env.action_space.n ** (1/n_agents))

        # n_actions = env.get_num_actions() #env.action_space.n
        n_actions = self.env.action_space.n
        self.state, _ = self.env.reset()
        self.n_observations = len(self.state)  # Why don't get env.observation_space.box (Generalize to discrete and continuous environmen)

        if 'linear' in nn:
            self.policy_net = Network(self.n_observations, n_actions, hid_dim)
            self.target_net = Network(self.n_observations, n_actions, hid_dim)
        elif 'CNN' in nn:
            self.policy_net = CNN(self.state.shape[-1], n_actions)
            self.target_net = CNN(self.state.shape[-1], n_actions)  
        elif 'duel' in nn:
            self.policy_net = DuelNetwork(self.n_observations, n_actions, hid_dim)
            self.target_net = DuelNetwork(self.n_observations, n_actions, hid_dim)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

        if 'per' in self.buffer:
            alpha = 0.7
            self.memory = PrioritizeExperienceReplay(capacity, alpha, self.env)

        elif 'simple' in self.buffer:
            self.memory = ReplayMemory(capacity=capacity)

        # If gpu is to be used
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.steps_done = 0

        # Variables to collect statistics
        self.writer = SummaryWriter(log_dir)
        self.loss = torch.zeros(1)
        self.cum_reward = deque([], maxlen=100)
        self.cum_discounted_reward = deque([], maxlen=100)
        self.max_cum_reward = 0
        self.max_cum_discounted_reward = 0
        


    def select_action(self, state):
        self.eps_threshold = 0 if self.evaluate_flag else self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)

        if random.random() > self.eps_threshold:
            with torch.no_grad():
                # t.max(1) --> Largest column value of each row.
                # [1] Second column on max --> Index where max element was found, thus we pick the action that maximizes reward
                # view(1, 1) --> Reshape a tensor to have torch.Size([1, 1]) instead of torch.Size([])
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                # return self.policy_net(state).max(1)[1].view(1, 1)
                return self.policy_net(state).max(1)[1].view(1, 1).item()
        else:
            # return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.float32)
            # return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.int64)
            return self.env.action_space.sample()
            # return torch.tensor([[np.random.randint(self.env.get_num_actions())]], device=self.device, dtype=torch.long)
   
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        if 'per' in self.buffer:
            # I need to understand what is a good beta
            samples = self.memory.sample(self.batch_size)

            state_batch = torch.tensor(samples['obs'], dtype=torch.float32, device=self.device)
            action_batch = torch.tensor(samples['action'], device=self.device, dtype=torch.long).unsqueeze(1)
            reward_batch = torch.tensor(samples['reward'], device=self.device)
            next_state_batch = torch.tensor(samples['next_obs'], dtype=torch.float32, device=self.device)


            non_final_mask = torch.tensor([s is not None for s in next_state_batch])
            non_final_next_states = torch.cat([s for s in next_state_batch.unsqueeze(1) if s is not None])
            
            transitions = samples['weights']
            indexes = samples['indexes']

        elif 'simple' in self.buffer:
            transitions = self.memory.sample(self.batch_size)
            # Converts batch-array of Transitions to Transitons of batch-arrays
            batch = Transition(*zip(*transitions))

            # Convert to tensors
            batch = Transition(state=torch.stack([torch.from_numpy(s) for s in batch.state]),
                            action=torch.stack([torch.from_numpy(a) for a in batch.action]),
                            next_state=torch.stack([torch.from_numpy(n) for n in batch.next_state]),
                            reward=torch.stack([torch.from_numpy(r) for r in batch.reward]),
                            )

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

        # Update priorities with TD_error
        if 'per' in self.buffer:
            priorities = (expected_state_action_values.unsqueeze(1) - state_action_values).abs() + 1e-5 
            self.memory.update_priorities(indexes, priorities)
        

    def train(self, num_episodes):
        if torch.cuda.is_available():
            num_episodes *= 10

        for i_episode in range(num_episodes):
            state, info = self.env.reset()
            # state = self.env.reset()
            # state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            loss_mean = 0
            cum_reward = 0
            cum_discounted_reward = 0
            cum_gamma = self.gamma
            for t in count():
                action = self.select_action(state)
                # observation, reward, terminated, truncated, _ = self.env.step(action.item())
                observation, reward, terminated, truncated, _ = self.env.step(action)
                # observation, reward, terminated = self.env.step(action.item())
                cum_reward += reward
                cum_discounted_reward += cum_gamma * reward
                cum_gamma *= self.gamma
                # reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                # next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                next_state = None if terminated else observation

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                loss_mean += (self.loss.item() - loss_mean) / (t + 1)

                # Should I update the model after every single step?
                # Soft update of the target network's weights ( Why not use torch.nn.utisl.soft_update() in the parameters?)
                self.update_network()
                # target_net_state_dict = self.target_net.state_dict()
                # policy_net_state_dict = self.policy_net.state_dict()
                # for key in policy_net_state_dict:
                #     target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                # self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.steps_done += 1
                    self.cum_reward.append(cum_reward)
                    self.cum_discounted_reward.append(cum_discounted_reward)

                    
                    # if len(self.cum_discounted_reward) == self.cum_discounted_reward.maxlen and np.mean(self.cum_discounted_reward) > self.max_cum_discounted_reward:
                    #     self.max_cum_discounted_reward = np.mean(self.cum_discounted_reward)
                    #     torch.save(self.policy_net.state_dict(), os.path.join(self.log_dir, 'my_model.pth'))
                    #     self.evaluate(50, i_episode, self.policy_net)
                    # if i_episode % 1000 == 0:
                    #     self.evaluate(50, i_episode, self.policy_net)


                    self.writer.add_scalar('loss', loss_mean, i_episode) 
                    self.writer.add_scalar('episode_len', t+1, i_episode)
                    self.writer.add_scalar('reward', cum_reward, i_episode)
                    self.writer.add_scalar('disc_reward', cum_discounted_reward, i_episode)
                    self.writer.add_scalar('epsilon', self.eps_threshold, i_episode)

                    # Saving and evaluating the model
                    self.save_evaluate_model(i_episode)
                    break
        self.writer.close()
        

    def soft_update(self, policy, target):
        target_net_state_dict = target
        policy_net_state_dict = policy
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        return target_net_state_dict

    def update_network(self):
        self.target_net.load_state_dict(self.soft_update(self.policy_net.state_dict(), self.target_net.state_dict()))

    def save_evaluate_model(self, i_episode):
        # Saving and evaluating the model
        if len(self.cum_discounted_reward) == self.cum_discounted_reward.maxlen and np.mean(self.cum_discounted_reward) > self.max_cum_discounted_reward:
            self.max_cum_discounted_reward = np.mean(self.cum_discounted_reward)
            torch.save(self.policy_net.state_dict(), os.path.join(self.log_dir, 'my_model.pth'))
            self.evaluate(50, i_episode, self.policy_net)
        if i_episode % 1000 == 0:
            self.evaluate(50, i_episode, self.policy_net)


    def evaluate(self, num_episodes, episode_number, model):
        self.evaluate_flag = True
        history_cum_reward = []
        history_cum_disc_reward = []
        for i_episode in range(num_episodes):
            state, info = self.env.reset()
            # state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            cum_reward = 0
            cum_discounted_reward = 0
            cum_gamma = self.gamma
            for t in count():
                action = self.select_action(state)
                # with torch.no_grad():
                #     action = model(state).max(1)[1].view(1, 1)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                # observation, reward, terminated = self.env.step(action.item())
                cum_reward += reward
                cum_discounted_reward += cum_gamma * reward
                cum_gamma *= self.gamma
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Move to the next state
                state = next_state

                if done:
                    history_cum_reward.append(cum_reward)
                    history_cum_disc_reward.append(cum_discounted_reward)
                    break
        self.writer.add_scalar('reward/evaluated', np.mean(history_cum_reward), episode_number)
        self.writer.add_scalar('disc_reward/evaluated', np.mean(cum_discounted_reward), episode_number)
        self.evaluate_flag = False



class AgentOneAtTime(Agent):
    
    def __init__(self, env, batch_size, gamma, eps_start, eps_end, eps_decay, tau, 
                 lr, hid_dim=128, capacity=10_000, alg=['dqn'], log_dir='logs/', nn=['linear'], n_agents=2):
        super().__init__(env, batch_size, gamma, eps_start, eps_end, eps_decay, tau, 
                        lr, hid_dim, capacity, alg, log_dir, nn)


        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.alg = alg
        self.log_dir = log_dir
        self.n_agents = n_agents
        # Networks for the extended agents 1 and 2
        # Network has extended input n_observations + n_actions (Consider a1 and a2 have the same dimensions)
        self.policy_net = [None] * n_agents
        self.target_net = [None] * n_agents
        self.optimizer = [None] * n_agents
        self.loss = [None] * n_agents

        self.n_actions = int(env.action_space.n ** (1/n_agents))
       
        for i in range(n_agents):

            self.policy_net[i] = Network(self.n_observations + i, self.n_actions, hid_dim)
            self.target_net[i] = Network(self.n_observations + i, self.n_actions, hid_dim)
            self.target_net[i].load_state_dict(self.policy_net[i].state_dict())
            self.optimizer[i] = optim.AdamW(self.policy_net[i].parameters(), lr=lr, amsgrad=True)
            self.loss[i] = torch.zeros(1)


    def select_action(self, state):
        self.eps_threshold = 0 if self.evaluate_flag else self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        # eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)

        if random.random() > self.eps_threshold:
            with torch.no_grad():
                # t.max(1) --> Largest column value of each row.
                # [1] Second column on max --> Index where max element was found, thus we pick the action that maximizes reward
                # view(1, 1) --> Reshape a tensor to have torch.Size([1, 1]) instead of torch.Size([])
                action1 = self.policy_net[0](state).max(1)[1].view(1, 1)
                state_action1 = torch.cat((state, action1), dim=1)
                action2 = self.policy_net[1](state_action1).max(1)[1].view(1, 1)
                action = action1 + self.n_actions * action2

                return action
        else:

            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)    


    def extend_state(self, state_batch, action_batch):
        # Extract first and second values from dictionary
        action1_batch = torch.tensor([self.env.action_dict[key.item()][0] for key in action_batch.flatten()])       
        action2_batch = torch.tensor([self.env.action_dict[key.item()][1] for key in action_batch.flatten()])

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


        # --- Core of the DQN algorithm -------
        state_action1_values = self.policy_net[0](state_batch).gather(1, action1_batch)
        state_action1_action2_values = self.policy_net[1](state_action1_batch).gather(1, action2_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)


        if 'dqn' in self.alg:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net[0](non_final_next_states).max(1)[0]
                expected_state_action1_values = self.target_net[1](state_action1_batch).max(1)[0]

        if 'ddqn' in self.alg:
            # next_action_values = torch.zeros(self.batch_size, device=self.device)
            with torch.no_grad():
                next_action1_values = self.policy_net[0](non_final_next_states).argmax(1).view(-1, 1)
                next_state_values[non_final_mask] = self.target_net[0](non_final_next_states).gather(1, next_action1_values).view(-1)

                action2_values = self.policy_net[1](state_action1_batch).argmax(1).view(-1, 1)
                expected_state_action1_values = self.target_net[1](state_action1_batch).gather(1, action2_values).view(-1)



        expected_state_action1_action2_values = (next_state_values * self.gamma) + reward_batch
        # --- Core of the DQN algorithm -------

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        self.loss[0] = criterion(state_action1_values, expected_state_action1_values.unsqueeze(1))
        self.loss[1] = criterion(state_action1_action2_values, expected_state_action1_action2_values.unsqueeze(1))  

        for i in range(self.n_agents):
            # Optimize the model
            self.optimizer[i].zero_grad()
            self.loss[i].backward()

            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net[i].parameters(), 100)
            self.optimizer[i].step()


    def train(self, num_episodes):
        if torch.cuda.is_available():
            num_episodes *= 10

        for i_episode in range(num_episodes):
            state, info = self.env.reset()
            # state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            loss_mean = [0] * self.n_agents
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
                for i in range(self.n_agents):
                    loss_mean[i] += (self.loss[i].item() - loss_mean[i]) / (t + 1)

                # Should I update the model after every single step?
                # Soft update of the target network's weights ( Why not use torch.nn.utisl.soft_update() in the parameters?)
                self.update_network()


                if done:
                    self.steps_done += 1
                    self.cum_reward.append(cum_reward)
                    self.cum_discounted_reward.append(cum_discounted_reward)

                    # # Saving and evaluating the model
                    # self.save_evaluate_model(i_episode)

                    for i in range(self.n_agents):
                        self.writer.add_scalar(f'loss/loss{i}', loss_mean[i], i_episode) 
                    self.writer.add_scalar('episode_len', t+1, i_episode)
                    self.writer.add_scalar('reward', cum_reward, i_episode)
                    self.writer.add_scalar('disc_reward', cum_discounted_reward, i_episode)
                    self.writer.add_scalar('epsilon', self.eps_threshold, i_episode)

                    # Saving and evaluating the model
                    self.save_evaluate_model(i_episode)
                    break
        self.writer.close()

    def update_network(self):
        for i in range(self.n_agents):
        # Soft update of the target network's weights ( Why not use torch.nn.utisl.soft_update() in the parameters?)
            self.target_net[i].load_state_dict(self.soft_update(self.policy_net[i].state_dict(), self.target_net[i].state_dict()))

    def save_evaluate_model(self, i_episode):
        # Saving and evaluating the model
        if len(self.cum_discounted_reward) == self.cum_discounted_reward.maxlen and np.mean(self.cum_discounted_reward) > self.max_cum_discounted_reward:
            self.max_cum_discounted_reward = np.mean(self.cum_discounted_reward)
            for i in range(self.n_agents):
                torch.save(self.policy_net[i].state_dict(), os.path.join(self.log_dir, f'my_model_{i}.pth'))
            self.evaluate(50, i_episode, self.policy_net)
        if i_episode % 1000 == 0:
            self.evaluate(50, i_episode, self.policy_net)


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
        alg = ['ddqn']
        num_episodes = 50
        num_trial = 3
        env_name = 'four-room-multiagent-v0'
        num_agents = 1  

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
        # is_ipython = 'inline' in matplotlib.get_backend()
        # if is_ipython:
        #     from IPython import display

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
        
        unique_id = datetime.now().strftime("%Y_%m_%d__%H_%M_%S__%f")[:-4]
        name = f'logs/{env_name}_nag{num_agents}_{alg[0]}_nt{num_trial:03}_run_{unique_id}'
        system_name = platform.system()
        if platform.system() == 'Linux':
            log_dir = f'/data/p285087/drl_alg/{name}'
        else:
            log_dir = name


        my_dqn = Agent(env, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, hid_dim=hid_dim, capacity=capacity, alg=alg, log_dir=log_dir)
        my_dqn.train(num_episodes)

        # print('Complete')
        # my_dqn.plot_durations(show_result=True, save=True)
        # plt.ioff()
        # plt.show()


