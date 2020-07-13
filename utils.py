import numpy as np
import torch
from time import time
import random

from torch.utils.data import Dataset

def safe_update_dict(old_dict, new_dict):
    for k in new_dict:
        if k in old_dict:
            old_dict[k] = new_dict[k]
        else:
            raise Exception("Update dict have unknown keys: ",k)

class NonSequentialDataset(Dataset):
    def __init__(self, *arrays):
        super().__init__()
        # self.arrays = [array.reshape(-1, *array.shape[2:]) for array in arrays]
        self.arrays = [array for array in arrays]

    def __getitem__(self, index):
        return [array[index] for array in self.arrays]

    def __len__(self):
        return len(self.arrays[0])


class RunningStats(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.std = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / (self.count + batch_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        self.mean = new_mean
        self.var = new_var
        self.std = np.maximum(np.sqrt(self.var), 1e-6)
        self.count = batch_count + self.count


def discount_horizon(rewards, is_terminals, discount, device, horizon):
    res = []
    discounted_reward = 0
    horizon_counter = 0
    for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
        if horizon_counter % horizon == 0:
            discounted_reward = 0
            horizon_counter = 0
        horizon_counter += 1
        if is_terminal == 1:
            horizon_counter = 0
        discounted_reward = reward + discount * discounted_reward*(1-is_terminal)
        res.insert(0, discounted_reward)

    # Normalizing the rewards:
    res = torch.tensor(res).to(device)

    return res

def monte_carlo_reward_batch(rewards, is_terminals, discount, device):
    res = np.zeros_like(rewards)
    discounted_reward = 0
    for i in range(rewards.shape[1] - 1, -1, -1):
        discounted_reward = rewards[:, i] + discount * discounted_reward*(1-is_terminals[:, i])
        res[:,i] = discounted_reward

    return torch.from_numpy(res).to(device)

def monte_carlo_reward(rewards, is_terminals, discount, device):
    res = []
    discounted_reward = 0
    for i, (reward, is_terminal) in enumerate(zip(reversed(rewards), reversed(is_terminals))):
        discounted_reward = reward + discount * discounted_reward*(1-is_terminal)
        res.insert(0, discounted_reward)

    res = torch.tensor(res).to(device)

    return res

def GenerelizedAdvantageEstimate(gae_param, values, rewards, is_terminals, discount, device, horizon=None):
    assert(is_terminals[-1])
    if horizon is None:
        horizon = len(rewards)
    advantages = []
    cumulative_reward = []
    rewards = torch.tensor(rewards).to(device)
    is_terminals_tensor = torch.tensor(is_terminals).to(device).float()
    deltas = -values + rewards
    deltas[:-1] += (1-is_terminals_tensor[:-1])*discount*values[1:]
    running_advantage = 0
    running_reward = 0
    horizon_counter = 0
    for i, (reward, delta, is_terminal) in enumerate(zip(reversed(rewards), reversed(deltas), reversed(is_terminals))):
        if is_terminal or horizon_counter % horizon == 0:
            horizon_counter = 0
            running_advantage = 0
            running_reward = 0
        running_advantage = delta + discount*gae_param * running_advantage # * (1-is_terminal)
        running_reward = reward + discount * running_reward # * (1-is_terminal)
        advantages.insert(0, running_advantage)
        cumulative_reward.insert(0, running_reward)
        horizon_counter += 1

    # Normalizing the rewards:
    advantages = torch.tensor(advantages).to(device)
    cumulative_reward = torch.tensor(cumulative_reward).to(device)
    return advantages, cumulative_reward

class ListMemory:
    # Credit to Adrien Lucas Ecoffet
    def __init__(self, max_size):
        self.mem = [None]*max_size
        self.max_size = max_size
        self.next_index = 0
        self.size = 0
    
    def __len__(self):
        return self.size

    def add_sample(self, sample):
        self.mem[self.next_index] = sample
        self.size = min(self.max_size, self.size+1)
        self.next_index = (self.next_index + 1) % self.max_size

    def sample(self, batch_size, device):
        res = []
        batch_arrays = np.array(random.sample(self.mem[:self.size], k=batch_size), dtype=object)
        for i in range(batch_arrays.shape[1]):
            res += [torch.from_numpy(np.stack(batch_arrays[:, i])).to(device)]
        return tuple(res)


class FastMemory:
    "Pre alocates a numpy array for fast acess"
    def __init__(self, max_size, storage_sizes_and_types):
        self.max_size = max_size
        self.storages = []
        # self.prealocate_size = 1000 # Avoid too large memory allocation
        for s in storage_sizes_and_types:
            if type(s[0]) == int:
                shape = (self.max_size,s[0])
            else:
                shape = (self.max_size,) + s[0]
            self.storages += [np.zeros(shape, dtype=s[1])]

        self.next_index = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add_sample(self, sample):
        assert(len(sample) == len(self.storages))
        for i, s in enumerate(sample):
            self.storages[i][self.next_index] = s

        self.size = min(self.max_size, self.size+1)
        self.next_index = (self.next_index + 1) % self.max_size
        # if self.next_index >= self.storages[0].shape[0]:
        #     for i,s in enumerate(self.storages):
        #         added_shape = (min(self.max_size-self.size, self.prealocate_size),) + s.shape[1:]
        #         self.storages[i] = np.append(s, np.zeros(added_shape), axis=0)

    def sample(self, batch_size, device) :
        ind = np.random.randint(0, self.size, size=batch_size)
        batch = tuple([torch.from_numpy(storage[ind]).to(device) for storage in self.storages])

        return batch

    def clear_memory(self):
        for storage in self.storages:
            del storage[:]


class PrioritizedListMemory(ListMemory):
    def __init__(self, max_size, alpha=0.6):
        super().__init__(max_size)
        self.priorities = np.ones((max_size,), np.float32)
        self.alpha = alpha

    def add_sample(self, sample):
        super().add_sample(sample)
        self.priorities[self.next_index - 1] = self.priorities.max()

    def sample(self, batch_size, device, beta=0.4):
        probs  = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()


        res = []
        self.last_ind = np.random.choice(self.size, batch_size, p=probs)
        batch_arrays = np.array([self.mem[i] for i in self.last_ind], dtype=object) # TODO: OPTIMIZE
        for i in range(batch_arrays.shape[1]):
            res += [torch.from_numpy(np.stack(batch_arrays[:, i])).to(device)]
        res = tuple(res)

        # Compute sample weights
        weights = (self.size * probs[self.last_ind]) ** (-beta)
        weights /= weights.max()

        return res + (weights,)

    def update_priorities(self, batch_priorities):
        self.priorities[self.last_ind] = batch_priorities

class PrioritizedMemory(FastMemory):
    def __init__(self, max_size, storage_sizes_and_types, alpha=0.6):
        super().__init__(max_size, storage_sizes_and_types)
        self.priorities = np.ones((max_size,), np.float32)
        self.alpha = alpha

    def add_sample(self, sample):
        super().add_sample(sample)
        self.priorities[self.next_index - 1] = self.priorities.max()

    def sample(self, batch_size, device, beta=0.4):
        probs  = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()

        ind = np.random.choice(self.size, batch_size, p=probs)
        self.last_ind = ind
        batch = tuple([torch.from_numpy(storage[ind]).to(device) for storage in self.storages])

        # Compute sample weights
        weights = (self.size * probs[ind]) ** (-beta)
        weights /= weights.max()

        return batch + (weights,)

    def update_priorities(self, batch_priorities):
        self.priorities[self.last_ind] = batch_priorities


def update_net(model_to_change, reference_model, tau):
    for target_param, local_param in zip(model_to_change.parameters(), reference_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def measure_time(fun, *args):
    s = time()
    for i in range(10):
        fun(*args)
    print("time: ", (time() - s) / 10)