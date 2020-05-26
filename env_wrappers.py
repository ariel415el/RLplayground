import gym
import cv2
from gym.wrappers.pixel_observation import PixelObservationWrapper
# from gym.wrappers.atari_preprocessing import AtariPreprocessing
from _collections import deque
import numpy as np

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
    def __init__(self, env, target_shape=(80,80)):
        super().__init__(env)
        self.target_shape = target_shape

    def _preprocess_image(self, img):
        img = img.mean(axis=2)
        img = cv2.resize(img, self.target_shape)
        return img

    def reset(self):
        state = self.env.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = self._preprocess_image(state)
        return state, reward, done, None

class state_stack_wrapper(EnvDiligator):
    def __init__(self, env, stack_size=2, default_action=0):
        super().__init__(env)
        self.default_action = default_action
        self.stack_size = stack_size
        self.last_states = deque(maxlen=self.stack_size)

    def reset(self):
        self.last_states.clear()
        state = self.env.reset()
        self.last_states.append(state)
        while len(self.last_states) < self.stack_size:
            state, reward, done, info = self.env.step(self.default_action)
            self.last_states.append(state)
        return np.array(self.last_states).transpose(1,2,0)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
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

class PLE2GYM_wrapper(object):
    def __init__(self):
        import ple
        import os
        from ple.games.flappybird import FlappyBird
        os.environ['SDL_VIDEODRIVER'] = 'dummy' # avoid opening screen
        self.ple_game = ple.PLE(FlappyBird(), display_screen=False, force_fps=False)
        self.ple_game.init()
        self.allowed_actions = self.ple_game.getActionSet()
        self.state_keys = [k for k in self.ple_game.getGameState()]
        self._max_episode_steps = 100000

    def reset(self):
        self.ple_game.reset_game()
        state = self.ple_game.getGameState()
        state = np.array([state[k] for k in self.state_keys], dtype=np.float32)
        return state

    def step(self, action):
        reward = self.ple_game.act(self.allowed_actions[action])
        state = self.ple_game.getGameState()
        # state = np.array( [v for _, v in state.items()])
        state = np.array([state[k] for k in self.state_keys], dtype=np.float32)
        done = self.ple_game.game_over()
        return state, reward, done, None

    def render(self):
        self.ple_game.display_screen=True

    def seed(self, num):
        pass

    def close(self):
        pass

    def _get_image(self):
        image_rotated = np.fliplr(np.rot90(self.ple_game.getScreenRGB(),3)) # Hack to fix the rotated image returned by ple
        return image_rotated