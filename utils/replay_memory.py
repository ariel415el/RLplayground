import random
import numpy as np
import torch

class ListMemory:
    # Credit to Adrien Lucas Ecoffet
    """A simple memory with variant number of records
        cycle replace old records
    """
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


class PrioritizedListMemory(ListMemory):
    """
    An implementation of the state memory introduced in the priorities replay memory paper
    Used with DQN variants only
    """
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