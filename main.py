import numpy as np
import os
import random
import loggers
import torch
from train_scripts.TrainConfigs import *
from train_scripts import train

if  __name__ == '__main__':
    SCORE_SCOPE=100
    LOG_FREQUENCY=20
    SEED=0
    TRAIN_ROOT="TEST_TRAINING"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # env_name, env_builder, agent, solved_score = solve_cart_pole("PPOParallel")
    # env_name, env_builder, agent, solved_score = solve_acrobot("DQN")
    # env_name, env_builder, agent, solved_score = solve_mountain_car("PPOParallel")
    # env_name, env_builder, agent, solved_score = solve_pendulum("PPO_2")
    # env_name, env_builder, agent, solved_score = solve_lunar_lander("PPO")
    # env_name, env_builder, agent, solved_score = solve_continous_lunar_lander("PPO")
    # env_name, env_builder, agent, solved_score = solve_bipedal_walker("PPO")
    # env_name, env_builder, agent, solved_score = solve_pong('DQN')
    env_name, env_builder, agent, solved_score = solve_breakout("PPOParallel")
    # env_name, env_builder, agent, solved_score = solve_2d_walker("PPO")
    # env_name, env_builder, agent, solved_score = solve_ant("PPO")
    # env_name, env_builder, agent, solved_score = solve_humanoid()
    # env_name, env_builder, agent, solved_score = solve_half_cheetah()
    # env_name, env_builder, agent, solved_score = solve_super_mario("PPO")

    # env.seed(SEED)

    # Train
    train_dir = os.path.join(TRAIN_ROOT, env_name,  agent.name)
    os.makedirs(train_dir, exist_ok=True)

    logger = loggers.plt_logger(log_frequency=LOG_FREQUENCY, logdir=train_dir)
    # logger = loggers.TB_logger(k=SCORE_SCOPE, log_frequency=LOG_FREQUENCY, logdir=train_dir)
    # logger = loggers.logger(k=SCORE_SCOPE, log_frequency=LOG_FREQUENCY, logdir=train_dir)

    agent.set_reporter(logger)
    progress_maneger = train.train_progress_manager(train_dir, solved_score, SCORE_SCOPE, logger, checkpoint_steps=0.2, train_episodes=1000000, temporal_frequency=60**2)
    #
    train.train_agent_multi_env(env_builder, agent, progress_maneger, test_frequency=250, test_episodes=1, save_videos=True)
    # train.train_agent(env_builder, agent, progress_maneger, test_frequency=250, test_episodes=1, save_videos=True)


    # # Test
    # render=True
    # if render:
    #     from pyglet.gl import *  # Fixes rendering issues of openAi gym with wrappers
    # score = train.test(env, agent, 3, render=render, delay=0.025)
    # print("Reward over %d episodes: %f"%(3, score))
