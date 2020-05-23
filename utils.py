import numpy as np
import torch
from time import time
import gym
import cv2
# from ple.games.pong import Pong
# from ple.games.pixelcopter import Pixelcopter
# from ple import PLE
from gym.wrappers.pixel_observation import PixelObservationWrapper
# from gym.wrappers.atari_preprocessing import AtariPreprocessing
from _collections import deque
class EnvDiligator(object):
    def __init__(self, env):
        self.env = env
        self._max_episode_steps = self.env._max_episode_steps

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def seed(self, num):
        self.env.seed(num)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


class image_preprocess_wrapper(EnvDiligator):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self._max_episode_steps = env._max_episode_steps
        self.default_action = 0
        self.state_depth = 4
        self.last_states = deque(maxlen=self.state_depth)
        self.target_shape = (80, 105)

    def preprocess_image(self, img):
        img = img.mean(axis=2)
        img = cv2.resize(img, self.target_shape)
        return img

    def reset(self):
        self.last_states.clear()
        state = self.env.reset()
        state = self.preprocess_image(state)
        self.last_states.append(state)
        while len(self.last_states) < self.state_depth:
            state, reward, done, info = self.env.step(self.default_action)
            state = self.preprocess_image(state)
            self.last_states.append(state)

        return np.array(self.last_states).transpose(1,2,0)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = self.preprocess_image(state)
        self.last_states.append(state)
        return np.array(self.last_states).transpose(1,2,0), reward, done, None


class my_image_level_wrapper(EnvDiligator):
    def __init__(self, env):
        super().__init__(env)
        self.env = PixelObservationWrapper(env)

    def reset(self):
        return self.env.reset()['pixels']

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = state['pixels']
        return state, reward, done, info

# class PLE2GYM_wrapper(object):
#     def __init__(self, render=False):
#         self.ple_game = PLE(Pixelcopter(), fps=30, display_screen=render, force_fps=False)
#         self.ple_game.init()
#         self.allowed_actions = self.ple_game.getActionSet()
#         self.state_keys = ['player_y', 'player_vel', 'player_dist_to_ceil', 'player_dist_to_floor', 'next_gate_dist_to_player', 'next_gate_block_top', 'next_gate_block_bottom']
#         self._max_episode_steps = 100000
#
#     def reset(self):
#         self.ple_game.reset_game()
#         state = self.ple_game.getGameState()
#         state = np.array([state[k] for k in self.state_keys])
#         return state
#
#     def step(self, action):
#         reward = self.ple_game.act(self.allowed_actions[action])
#         state = self.ple_game.getGameState()
#         # state = np.array( [v for _, v in state.items()])
#         state = np.array([state[k] for k in self.state_keys])
#         done = self.ple_game.game_over()
#         return state, reward, done, None
#
#     def seed(self, num):
#         pass
#
#     def close(self):
#         pass


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

def update_net(model_to_change, reference_model, tau):
    for target_param, local_param in zip(model_to_change.parameters(), reference_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def measure_time(fun, *args):
    s = time()
    for i in range(10):
        fun(*args)
    print("time: ", (time() - s) / 10)