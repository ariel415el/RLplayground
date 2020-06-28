import cv2
from time import time, sleep
import os
import gym_minigrid

def _test_ple():
    from ple.games.pong import Pong
    from ple.games.flappybird import FlappyBird
    from ple import PLE
    # os.environ['SDL_VIDEODsRIVER'] = 'dummy'
    game = Pong()
    game = FlappyBird()
    ple_game = PLE(game, fps=30, display_screen=True)
    ple_game.init()
    ALLOWED_ACTIONS = ple_game.getActionSet()

    print(ALLOWED_ACTIONS)
    action = 0
    start = time()
    t = 0
    while True:
        ep_reward = 0
        ple_game.reset_game()
        while not ple_game.game_over():
            sleep(0.1)
            t += 1
            if t % 15== 5:
                action = 0
            else: action = 1
            reward = ple_game.act(ALLOWED_ACTIONS[action])
            # print(reward)
            ep_reward += reward
        print(ep_reward, t, t / (time() - start))


def test_gym():
    from time import sleep
    import gym
    import pybulletgym  # register PyBullet enviroments with open ai gym
    # import gym_ple
    from gym.wrappers.pixel_observation import PixelObservationWrapper
    from gym.wrappers.atari_preprocessing import AtariPreprocessing
    # env = gym.make("BreakoutNoFrameskip-v4")
    # env = gym.make("BreakoutDeterministic-v4")
    # env = gym.make("HumanoidPyBulletEnv-v0")
    env = gym.make('MiniGrid-Dynamic-Obstacles-16x16-v0')
    # env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)  # Get pixel observations
    env = gym_minigrid.wrappers.ImgObsWrapper(env) # Get rid of the 'mission' field

    # env = gym.make("PongNoFrameskip-v0")
    # env = gym.make("FlappyBird-v0")
    # env = PixelObservationWrapper(env, pixels_only=True)
    print(env.spec.id)
    # env = AtariPreprocessing(env)
    done = False
    t = 0
    start = time()
    # print(env.action_space, state.shape, env.game_state.getGameState());exit()
    while True:
        total_reward = 0
        state = env.reset()
        done = False
        # import pdb;pdb.set_trace()
        # print(env.action_space, state.shape, env.ple.getGameState());import cv2; cv2.imwrite("asd.png", state);exit()
        steps = 0
        while not done:
            t += 1
            # env.render()
            print(state.shape)
            # sleep(0.001)
            img = env.render(mode='rgb_array')
            cv2.imshow("preview", state)
            k = cv2.waitKey(0)
            cv2.destroyAllWindows()
            action = env.action_space.sample()
            # action = 1
            # print(action)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps+=1
        # print(steps)
        # print(t, t / (time() - start))

if __name__ == '__main__':
    test_gym()
    # _test_ple()