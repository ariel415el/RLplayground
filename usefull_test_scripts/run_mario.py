from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import cv2
import numpy as np
from matplotlib import pyplot as plt

env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# print(COMPLEX_MOVEMENT)
# print(SIMPLE_MOVEMENT)
print(env.action_space, env.observation_space)
done = True
r = 0
for step in range(5000):
    if done:
        print("Episode_reward", r)
        r = 0
        print('reset')
        state = env.reset()
    # state, reward, done, info = env.step(env.action_space.sample())
    state, reward, done, info = env.step(1 + (step%2))
    r += reward
    env.render()


env.close()