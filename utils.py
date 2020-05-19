import numpy as np
import torch

def update_net(model_to_change, reference_model, tau):
    for target_param, local_param in zip(model_to_change.parameters(), reference_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class FastMemory:
    def __init__(self, max_size, storage_sizes_and_types):
        self.max_size = max_size
        self.storages = []
        for s in storage_sizes_and_types:
            if type(s) == int:
                s = (s, np.float32)
            self.storages += [np.zeros(shape=(max_size, s[0]), dtype=s[1])]

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
