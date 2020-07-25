import numpy as np
import torch
import random
from torch.utils.data import Dataset


def safe_update_dict(old_dict, new_dict):
    """Update dict only with keys it already has"""
    for k in new_dict:
        if k in old_dict:
            old_dict[k] = new_dict[k]
        else:
            raise Exception("Update dict have unknown keys: ",k)


class BasicDataset(Dataset):
    """Createds a torch Dataset from list of any arrays"""
    def __init__(self, *arrays):
        super().__init__()
        self.arrays = [array for array in arrays]

    def __getitem__(self, index):
        return [array[index] for array in self.arrays]

    def __len__(self):
        return len(self.arrays[0])


class RunningStats(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
    def __init__(self, shape, computation_module=np, epsilon=1e-4):
        self.computation_module = computation_module
        self.mean = computation_module.zeros(shape, dtype=float)
        self.var = computation_module.ones(shape, dtype=float)
        self.std = computation_module.ones(shape, dtype=float)
        self.count = epsilon

    def scale(self, arr):
        return (arr - self.mean) / self.std

    def update(self, x):
        batch_mean = x.mean(0)
        batch_var = x.var(0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

        if self.computation_module == np:
            self.std = np.maximum(np.sqrt(self.var), 1e-6)
        else:
            self.std = torch.clamp(torch.sqrt(self.var), 1e-6, np.inf)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / (self.count + batch_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        self.mean = new_mean
        self.var = new_var
        self.count = batch_count + self.count


def discount(rewards, is_terminals, discount, device):
    """Discount a serie of rewards by a discount factor"""
    res = []
    discounted_reward = 0
    for i, (reward, is_terminal) in enumerate(zip(reversed(rewards), reversed(is_terminals))):
        discounted_reward = reward + discount * discounted_reward*(1-is_terminal)
        res.insert(0, discounted_reward)

    res = torch.tensor(res).to(device)

    return res


def discount_batch(rewards: torch.tensor, is_terminals: torch.tensor, discount: torch.tensor, device):
    res = torch.zeros_like(rewards).to(device)
    discounted_reward = 0
    for i in range(rewards.shape[1] - 1, -1, -1):
        discounted_reward = rewards[:, i] + discount * discounted_reward*(1-is_terminals[:, i])
        res[:,i] = discounted_reward

    return res


def discount_horizon(rewards, is_terminals, discount, device, horizon):
    """Discount and zero the sum every 'horizon' steps"""
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




def update_net(model_to_change, reference_model, tau):
    """
    Update weights with tau factor
    :param tau: update weight
    """
    for target_param, local_param in zip(model_to_change.parameters(), reference_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

