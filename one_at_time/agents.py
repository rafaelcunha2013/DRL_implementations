import math
import random

from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from memory import ReplayMemory
from network import Network

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

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
                loss_mean += (self.loss.item() - loss_mean) / (t + 1)


                # Soft update of the target network's weights ( Why not use torch.nn.utisl.soft_update() in the parameters?)
                # target_net_state_dict = self.target_net.state_dict()
                # policy_net_state_dict = self.policy_net.state_dict()
                # for key in policy_net_state_dict:
                #     target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                # self.target_net.load_state_dict(target_net_state_dict)

                self.target_net.load_state_dict(self.soft_update(self.policy_net.state_dict(), self.target_net.state_dict()))


                if done:
                    self.writer.add_scalar('loss', loss_mean, i_episode) 
                    self.writer.add_scalar('episode_len', t+1, i_episode)
                    self.writer.add_scalar('reward', cum_reward, i_episode)
                    self.writer.add_scalar('disc_reward', cum_discounted_reward, i_episode)
                    break
        self.writer.close()

    
    def soft_update(self, policy, target):
        target_net_state_dict = target
        policy_net_state_dict = policy
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        return target_net_state_dict



class AgentOneAtTime:

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
                self.target_net1.load_state_dict(self.soft_update(self.policy_net1.state_dict(), self.target_net1.state_dict()))
                self.target_net2.load_state_dict(self.soft_update(self.policy_net2.state_dict(), self.target_net2.state_dict()))


                if done:
                    self.writer.add_scalar('loss1', loss_mean1, i_episode) 
                    self.writer.add_scalar('loss2', loss_mean2, i_episode) 
                    self.writer.add_scalar('episode_len', t+1, i_episode)
                    self.writer.add_scalar('reward', cum_reward, i_episode)
                    self.writer.add_scalar('disc_reward', cum_discounted_reward, i_episode)
                    break
        self.writer.close()


    def soft_update(self, policy, target):
        target_net_state_dict = target
        policy_net_state_dict = policy
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        return target_net_state_dict





