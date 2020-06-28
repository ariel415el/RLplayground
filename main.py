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

    # env_name, env, agent, solved_score = solve_cart_pole("PPO")
    # env_name, env, agent, solved_score = solve_acrobot("DQN")
    # env_name, env, agent, solved_score = solve_mountain_car("PPO_ICM")
    # env_name, env, agent, solved_score = solve_pendulum("PPO")
    # env_name, env, agent, solved_score = solve_lunar_lander()
    # env_name, env, agent, solved_score = solve_continous_lunar_lander("A2C")
    # env_name, env, agent, solved_score = solve_bipedal_walker("TD3")
    # env_name, env, agent, solved_score = solve_pong('DQN')
    # env_name, env, agent, solved_score = solve_breakout("DQN")
    # env_name, env, agent, solved_score = solve_2d_walker("PPO")
    # env_name, env, agent, solved_score = solve_ant("PPO")
    # env_name, env, agent, solved_score = solve_humanoid()
    # env_name, env, agent, solved_score = solve_half_cheetah()
    env_name, env, agent, solved_score = solve_super_mario("PPO")

    env.seed(SEED)

    # Train
    train_dir =  os.path.join(TRAIN_ROOT, env_name,  agent.name)
    os.makedirs(train_dir, exist_ok=True)
    assert(os.path.exists(train_dir))

    logger = loggers.plt_logger(k=SCORE_SCOPE, log_frequency=LOG_FREQUENCY, logdir=train_dir)
    # logger = train_logger.TB_logger(k=score_scope, log_frequency=10, logdir=train_dir)
    # logger = train_logger.logger(score_scope, log_frequency=10)

    agent.set_reporter(logger)
    train.train_agent(env, agent, train_dir, logger, solved_score=solved_score, test_frequency=100,
                      train_episodes=10000, test_episodes=1, save_videos=True, checkpoint_steps=0.2)

    # # Test
    # render=True
    # if render:
    #     from pyglet.gl import *  # Fixes rendering issues of openAi gym with wrappers
    # score = train.test(env, agent, 3, render=render, delay=0.025)
    # print("Reward over %d episodes: %f"%(3, score))
