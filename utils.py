import numpy as np
import torch
from time import time


from ple.games.pong import Pong
from ple.games.pixelcopter import Pixelcopter
from ple import PLE
def measure_time(fun, *args):
    s = time()
    for i in range(10):
        fun(*args)
    print("time: ", (time() - s) / 10)

class PLE2GYM_wrapper(object):
    def __init__(self, render=False):
        self.ple_game = PLE(Pixelcopter(), fps=30, display_screen=render, force_fps=False)
        self.ple_game.init()
        self.allowed_actions = self.ple_game.getActionSet()
        self._max_episode_steps = 100000

    def reset(self):
        self.ple_game.reset_game()
        state = self.ple_game.getGameState()
        state = [v for _, v in state.items()]
        return state

    def step(self, action):
        reward = self.ple_game.act(self.allowed_actions[action])
        state = self.ple_game.getGameState()
        # state = np.array( [v for _, v in state.items()])
        state = [v for _, v in state.items()]
        done = self.ple_game.game_over()
        return state, reward, done, None

    def seed(self, num):
        pass

    def close(self):
        pass

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

    def clear_memory(self):
        for storage in self.storages:
            del storage[:]

