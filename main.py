import numpy as np
import os
import random
from utils import loggers
import torch
from Train.TrainConfigs import *
from Train import train
from opt import *
if  __name__ == '__main__':
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # env_name, env_builder, agent, solved_score = prepare_cart_pole("PPOParallel")
    # env_name, env_builder, agent, solved_score = prepare_acrobot("DQN")
    # env_name, env_builder, agent, solved_score = prepare_mountain_car("PPOParallel")
    # env_name, env_builder, agent, solved_score = prepare_pendulum("PPO_2")
    env_name, env_builder, agent, solved_score = prepare_lunar_lander("PPOParallel")
    # env_name, env_builder, agent, solved_score = prepare_continous_lunar_lander("PPO")
    # env_name, env_builder, agent, solved_score = prepare_bipedal_walker("PPO")
    # env_name, env_builder, agent, solved_score = prepare_pong('DQN')
    # env_name, env_builder, agent, solved_score = prepare_breakout("PPOParallel")
    # env_name, env_builder, agent, solved_score = prepare_2d_walker("PPO")
    # env_name, env_builder, agent, solved_score = prepare_ant("PPO")
    # env_name, env_builder, agent, solved_score = prepare_humanoid()
    # env_name, env_builder, agent, solved_score = prepare_half_cheetah()
    # env_name, env_builder, agent, solved_score = prepare_super_mario("PPO")

    # env.seed(SEED)

    # Train
    train_dir = os.path.join(TRAIN_ROOT, env_name,  agent.name)
    os.makedirs(train_dir, exist_ok=True)

    logger = loggers.plt_logger(log_frequency=LOG_FREQUENCY, logdir=train_dir)
    # logger = loggers.TB_logger(log_frequency=LOG_FREQUENCY, logdir=train_dir)
    # logger = loggers.logger(log_frequency=LOG_FREQUENCY, logdir=train_dir)

    agent.set_reporter(logger)
    progress_maneger = train.train_progress_manager(train_dir, solved_score, SCORE_SCOPE, logger, checkpoint_steps=CKP_STEP, train_episodes=TRAIN_EPISODES, temporal_frequency=TEMPORAL_FREQ)
    #
    train.train_agent_multi_env(env_builder, agent, progress_maneger, test_frequency=TEST_FREQ, test_episodes=TEST_EPISODES, save_videos=TEST_EPISODES)
    # train.train_agent(env_builder, agent, progress_maneger, test_frequency=TEST_FREQ, test_episodes=TEST_EPISODES, save_videos=TEST_EPISODES)


    # # Test
    # render=True
    # if render:
    #     from pyglet.gl import *  # Fixes rendering issues of openAi gym with wrappers
    # score = train.test(env, agent, 3, render=render, delay=0.025)
    # print("Reward over %d episodes: %f"%(3, score))
