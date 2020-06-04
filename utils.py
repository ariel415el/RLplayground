import numpy as np
import torch
from time import time
from _collections import deque
import random

class ListMemory:
    # Credit to Adrien Lucas Ecoffet
    def __init__(self, max_size,storage_sizes_and_types):
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
        batch_arrays = np.array(random.sample(self.mem[:self.size], k=batch_size))
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