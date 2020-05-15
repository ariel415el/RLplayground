import numpy as np
import os
from time import time, sleep
import random
import gym
from descrete_agents import *
from continous_agents import *
import train_logger
import torch

from gym.wrappers.pixel_observation import PixelObservationWrapper

def train(env, actor, train_episodes, score_scope, solved_score, test_frequency=25):
    next_progress_checkpoint = 1
    # logger = TB_logger(score_scope, SummaryWriter(log_dir=os.path.join(TRAIN_DIR, "tensorboard_outputs",  actor.name)))
    logger = train_logger.plt_logger(score_scope, os.path.join(TRAIN_DIR,  actor.name))
    logger.log_test(test(env, actor, 3))
    for i in range(train_episodes):
        done = False
        state = env.reset()
        episode_rewards = []
        while not done:
            action = actor.process_new_state(state)
            state, reward, done, info = env.step(action)
            is_terminal = done and len(episode_rewards) < env._max_episode_steps
            actor.process_output(state, reward, is_terminal)
            episode_rewards += [reward]

        if (i+1) % test_frequency == 0:
            last_test_score = test(env, actor, 3)
            logger.log_test(last_test_score)
        last_k_scores = logger.log_train_episode(i, episode_rewards, actor.get_stats())
        if last_k_scores >= next_progress_checkpoint*0.2*solved_score:
            actor.save_state(os.path.join(TRAIN_DIR, actor.name + "_%.5f_weights.pt"%last_k_scores))
            next_progress_checkpoint += 1

        if last_k_scores > solved_score:
            print("Solved in %d episodes"%i)
            break

    actor.save_state(os.path.join(TRAIN_DIR, actor.name + "_final_weights.pt"))

    env.close()

def test(env,  actor, test_episodes=1, render=False):
    actor.train = False
    episodes_total_rewards = []
    for i in range(test_episodes):
        done = False
        state = env.reset()
        all_rewards = []
        while not done:
            if render:
                env.render()
            action = actor.process_new_state(state)
            state, reward, done, info = env.step(action)
            all_rewards += [reward]
        episodes_total_rewards += [np.sum(all_rewards)]
    score = np.mean(episodes_total_rewards)
    env.close()
    actor.train=True
    return score


if  __name__ == '__main__':
    # Choose enviroment
    # ENV_NAME="CartPole-v1"; s=4; a=2
    # ENV_NAME="LunarLander-v2"; s=8; a=4
    ENV_NAME="LunarLanderContinuous-v2";s=8; score_scope=99; solved_score=200
    # ENV_NAME="Pendulum-v0";s=3; score_scope=99; solved_score=-200
    # ENV_NAME="BipedalWalker-v3"; s=24; score_scope=99; solved_score=500
    # ENV_NAME="BipedalWalkerHardcore-v3"; s=24; score_scope=99; solved_score=300

    env = gym.make(ENV_NAME)

    # set seeds
    SEED=0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    env.seed(SEED)

    # Create agent
    NUM_EPISODES = 10000
    # actor = DQN_agent(s, a, NUM_EPISODES, train=True)
    # actor = vanila_policy_gradient_agent(s, a, NUM_EPISODES, train=True)
    # actor = actor_critic_agent(s, a, NUM_EPISODES, train=True, critic_objective="Monte-Carlo")
    # actor = actor_critic_agent(s, bounderies, NUM_EPISODES, train=True, critic_objective="Monte-Carlo")
    # actor = DDPG.DDPG(s, bounderies, NUM_EPISODES, train=True)
    actor = TD3.TD3(s, [env.action_space.low, env.action_space.high], NUM_EPISODES, train=True, action_space=env.action_space)
    # actor = PPO.PPO(s, bounderies, NUM_EPISODES, train=True)

    # env = PixelObservationWrapper(env)

    # Train
    os.makedirs("Training", exist_ok=True)
    TRAIN_DIR = os.path.join("Training", ENV_NAME)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    trained_weights = None
    # trained_weights = os.path.join(TRAIN_DIR, actor.name + "_trained_weights.pt")
    actor.load_state(trained_weights)

    # actor.hyper_parameters['exploration_steps'] = -1
    train(env, actor, NUM_EPISODES, score_scope, solved_score)

    # Test
    # score = test(env, actor, 3, render=True)
    # print("Reward over %d episodes: %f"%(3, score))
