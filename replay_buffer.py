import random
import numpy as np


class PrioritizeExperienceReplay:

    def __init__(self, capacity, alpha, env):
        self.capacity = capacity
        self.alpha = alpha

        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]
        self.max_priority = 1

        self.data = {
            'obs': np.zeros(shape=(capacity, env.observation_space.shape[0]), dtype=np.float32),
            'action': np.zeros(shape=capacity, dtype=np.int32),
            'reward': np.zeros(shape=capacity, dtype=np.float32),
            'next_obs': np.zeros(shape=(capacity, env.observation_space.shape[0]), dtype=np.float32)#,
            #'done': np.zeros(shape=capacity, dtype=np.bool)
        }

        # state, _ = env.reset()
        # state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        # observation, reward, terminated, truncated, _ = self.env.step(action.item())
        # n_actions = env.action_space.n

        # state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        # next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        # reward = torch.tensor([reward], device=self.device)
        # action = torch.tensor([action], device=self.device, dtype=torch.long)
        # done = torch.tensor([reward], device=self.device) 

        self.next_idx = 0
        self.size = 0

    def push(self, obs, action, next_obs, reward):

        idx = self.next_idx

        self.data['obs'][idx] = obs
        self.data['action'][idx] = action
        self.data['reward'][idx] = reward
        self.data['next_obs'][idx] = next_obs
        # self.data['done'][idx] = done

        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)
        priority_alpha = self.max_priority ** self.alpha
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx, priority_alpha):

        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])
        
    def _set_priority_sum(self, idx, priority):
        idx += self.capacity
        self.priority_sum[idx] = priority
        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        return self.priority_sum[1]
    
    def _min(self):
        return self.priority_min[1]
    
    def find_prefix_sum_idx(self, prefix_sum):
        idx = 1
        while idx < self.capacity:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        return idx - self.capacity
    
    def sample(self, batch_size, beta):
        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=batch_size, dtype=np.int32)
        }

        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx

        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = samples['indexes'][i]
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            weight = (prob * self.size) ** (-beta)
            samples['weights'][i] = weight / max_weight

        for k, v in self.data.items():
            samples[k] = v[samples['indexes']]

        return samples

    
    def update_priorities(self, indexes, priorities):

        for idx, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = priority ** self.alpha
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def is_full(self):
        return self.capacity == self.size
    
    def __len__(self):
        return self.size









