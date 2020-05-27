import numpy as np
import torch
from time import time


class FastMemory:
    def __init__(self, max_size, storage_sizes_and_types):
        self.max_size = max_size
        self.storages = []
        for s in storage_sizes_and_types:
            if type(s[0]) == int:
                shape = (max_size,s[0])
            else:
                shape = (max_size,) +s[0]
            self.storages += [np.zeros(shape, dtype=s[1])]

        self.next_index = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add_sample(self, sample):
        assert(len(sample) == len(self.storages))
        for i, s in enumerate(sample):
            self.storages[i][self.next_index] = s

        self.next_index = (self.next_index + 1) % self.max_size
        self.size = min(self.max_size, self.size+1)

    def sample(self, batch_size, device) :
        ind = np.random.randint(0, self.size, size=batch_size)
        # batch = [torch.from_numpy(storage[ind]).to(device).float() for storage in self.storages]
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