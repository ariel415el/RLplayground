from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import cv2
import numpy as np
from matplotlib import pyplot as plt

env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
# print(COMPLEX_MOVEMENT)
print(SIMPLE_MOVEMENT)
print(env.action_space, env.observation_space)
done = True
for step in range(5000):
    if done:
        print('reset')
        state = env.reset()
    # state, reward, done, info = env.step(env.action_space.sample())
    state, reward, done, info = env.step(0 + (step%2)*5)
    env.render()
    # import pdb;pdb.set_trace()
    # print(info);exit()
    # img = env.render('rgb_array')
    # img = state
    # img = np.mean(img,axis=2)
    # img = cv2.resize(img,(84,84))
    # plt.imshow(img)
    # plt.show()

env.close()