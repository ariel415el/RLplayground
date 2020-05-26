import numpy as np
import os
from time import time, sleep
import random
import gym
import gym_ple
from discrete_agents import *
from continous_agents import *
import train_logger
import torch
from utils import measure_time
from env_wrappers import PLE2GYM_wrapper

def train(env, actor, train_episodes, score_scope, solved_score, log_frequency=1, test_frequency=100):
    next_progress_checkpoint = 1
    next_test_progress_checkpoint = 1
    # logger = TB_logger(score_scope, SummaryWriter(log_dir=os.path.join(TRAIN_DIR, "tensorboard_outputs",  actor.name)))
    logger = train_logger.plt_logger(score_scope, log_frequency,  os.path.join(TRAIN_DIR,  actor.name))
    # logger = train_logger.logger(score_scope, log_frequency)
    logger.log_test(test(env, actor, 3))
    for i in range(train_episodes):
        done = False
        state = env.reset()
        episode_rewards = []
        while not done:
            # env.render()
            action = actor.process_new_state(state)
            state, reward, done, info = env.step(action)
            is_terminal = done and len(episode_rewards) < env._max_episode_steps
            # is_terminal = done
            actor.process_output(state, reward, is_terminal)
            episode_rewards += [reward]

        if (i+1) % test_frequency == 0:
            last_test_score = test(env, actor, 3)
            logger.log_test(last_test_score)
            if last_test_score >= next_test_progress_checkpoint * 0.2 * solved_score:
                actor.save_state(os.path.join(TRAIN_DIR, actor.name + "_test_%.5f_weights.pt" % last_test_score))
                next_test_progress_checkpoint += 1

        logger.update_train_episode(episode_rewards)
        if (i+1) % log_frequency == 0:
            last_k_scores = logger.output_stats(actor.get_stats())
            if last_k_scores >= next_progress_checkpoint*0.2*solved_score:
                actor.save_state(os.path.join(TRAIN_DIR, actor.name, actor.name + "_%.5f_weights.pt"%last_k_scores))
                next_progress_checkpoint += 1

            if last_k_scores > solved_score:
                print("Solved in %d episodes"%i)
                break

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


def get_env(seed):
    ##### gym discrete envs #####
    # env_name="CartPole-v1"; s=4; a=2;score_scope=100; solved_score=195
    # env_name="LunarLander-v2"; s=8; a=4; score_scope=20; solved_score=200
    # env_name="LunarLanderContinuous-v2";s=8; score_scope=100; solved_score=200
    # env_name="Pendulum-v0";s=3; score_scope=100; solved_score=-200
    # env_name="BipedalWalker-v3"; s=24; score_scope=100; solved_score=500
    # env_name="BipedalWalkerHardcore-v3"; s=24; score_scope=100; solved_score=300

    # image envs TODO
    # env_name='PixelChopper';s=7;a=2;score_scope=100; solved_score=100
    # env_name="LunarLander-v2"; s=(105,80); a=4; score_scope=20; solved_score=200
    # env_name="BreakoutNoFrameskip-v0"; s=(84,84, 4); a=3;score_scope=100; solved_score=195
    # env_name="BreakoutNoFrameskip-v0"; s=(84,84, 4); a=3;score_scope=100; solved_score=195

    # env = gym.make(env_name)
    # env = my_image_level_wrapper(env)
    # env = AtariPreprocessing(env)
    # env = image_preprocess_wrapper(env)

    # get_ple games

    env = PLE2GYM_wrapper()
    env_name = 'FlappyBird-ple';s = len(env.state_keys);a = len(env.allowed_actions);score_scope=100;solved_score=100



    env.seed(seed)
    return env, s, a, score_scope, solved_score, env_name


def get_agent(env, s, a):
    agent = DQN_agent.DQN_agent(s, a, train=True)
    # agent = DiscretePPO.PPO_descrete_action(s, a, train=True)
    # agent = vanila_policy_gradient_agent(s, a, train=True)
    # agent = actor_critic_agent(s, a, train=True, critic_objective="Monte-Carlo")
    # agent = actor_critic_agent(s, bounderies, train=True, critic_objective="Monte-Carlo")
    # agent = DDPG.DDPG(s, bounderies, train=True)
    # agent = TD3.TD3(s, [env.action_space.low, env.action_space.high], train=True, action_space=env.action_space)
    # agent = PPO.PPO(s, bounderies, train=True)
    return agent

if  __name__ == '__main__':

    SEED=0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env, s, a, score_scope, solved_score, env_name = get_env(SEED)
    agent = get_agent(env, s, a)

    # Train
    os.makedirs("Training", exist_ok=True)
    TRAIN_DIR = os.path.join("Training", env_name)
    os.makedirs(TRAIN_DIR, exist_ok=True)

    # actor.load_state(trained_weights)

    train(env, agent, 100000, score_scope, solved_score)

    # Test
    # actor.train = False
    # score = test(env, actor, 3, render=True)
    # print("Reward over %d episodes: %f"%(3, score))
