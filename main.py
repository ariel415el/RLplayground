import numpy as np
import os
from time import time, sleep
import random
import gym
from discrete_agents import *
from continous_agents import *
import train_logger
import torch
from ExternalAtariWrappers import get_final_env

MAX_TRAIN_EPISODES = 1000000


def train(env, actor, score_scope, solved_score, log_frequency=20, test_frequency=100):
    next_progress_checkpoint = 1
    next_test_progress_checkpoint = 1

    # Define loggers
    # logger = train_logger.TB_logger(score_scope, SummaryWriter(log_dir=os.path.join(TRAIN_DIR, "tensorboard_outputs",  actor.name)))
    logger = train_logger.plt_logger(score_scope, log_frequency,  os.path.join(TRAIN_DIR,  actor.name))
    # logger = train_logger.logger(score_scope, log_frequency)
    logger.log_test(test(env, actor, 3))
    for i in range(MAX_TRAIN_EPISODES):
        done = False
        state = env.reset()
        episode_rewards = []
        # Run a single episode
        while not done:
            # env.render()
            action = actor.process_new_state(state)
            state, reward, done, info = env.step(action)
            # define final test
            # is_terminal = done and len(episode_rewards) < env._max_episode_steps
            is_terminal = done
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
                # sleep(0.05)
            action = actor.process_new_state(state)
            state, reward, done, info = env.step(action)
            all_rewards += [reward]
        episodes_total_rewards += [np.sum(all_rewards)]
    score = np.mean(episodes_total_rewards)
    env.close()
    actor.train=True
    return score


# def get_env(seed):
#     ##### gym discrete envs #####
#     env_name="CartPole-v1"; s=4; a=2;score_scope=100; solved_score=195
#     # env_name="MountainCar-v0"; s=2; a=3;score_scope=100; solved_score=-110
#     # env_name="LunarLander-v2"; s=8; a=4; score_scope=20; solved_score=200
#     # env_name="LunarLanderContinuous-v2";s=8; score_scope=100; solved_score=200
#     # env_name="Pendulum-v0";s=3; score_scope=100; solved_score=-200
#     # env_name="BipedalWalker-v3"; s=24; score_scope=100; solved_score=500
#     # env_name="BipedalWalkerHardcore-v3"; s=24; score_scope=100; solved_score=300
#

#
# def get_agent(env, s, a):
#     agent = DQN_agent.DQN_agent(s, a, double_dqn=True, dueling_dqn=False, prioritized_memory=False, noisy_MLP=False)
#     # agent = DiscretePPO.PPO_descrete_action(s, a)
#     # agent = vanila_policy_gradient_agent(s, a)
#     # agent = actor_critic_agent(s, a, train=True, critic_objective="Monte-Carlo")
#     # agent = actor_critic_agent(s, bounderies, train=True, critic_objective="Monte-Carlo")
#     # agent = DDPG.DDPG(s, bounderies)
#     # agent = TD3.TD3(s, [env.action_space.low, env.action_space.high], train=True, action_space=env.action_space)
#     # agent = PPO.PPO(s, bounderies)
#     return agent


def solve_pendulum():
    env_name="Pendulum-v0";s=3; score_scope=100; solved_score=-200
    env = gym.make(env_name)
    hp = {'actor_lr':0.01, 'critic_lr':0.01, "exploration_steps":5000, "min_memory_for_learning":10000, "batch_size": 128}
    agent = TD3.TD3(s, env.action_space, [env.action_space.low, env.action_space.high], hp, train=True)
    return env_name, env, agent, score_scope, solved_score

def solve_bipedal_walker():
    env_name="BipedalWalker-v3"; s=24; score_scope=100; solved_score=500
    env = gym.make(env_name)
    # hp = {'lr':0.001, "min_playback":0, "max_playback":1000000, "update_freq": 100, 'hiden_layer_size':32, 'epsilon_decay':500}
    agent = TD3.TD3(s, env.action_space, [env.action_space.low, env.action_space.high], hp, train=True)
    return env_name, env, agent, score_scope, solved_score

def solve_cart_pole():
    env_name="CartPole-v1"; s=4; a=2;score_scope=100; solved_score=195
    env = gym.make(env_name)
    hp = {'lr':0.001, "min_playback":0, "max_playback":1000000, "update_freq": 100, 'hiden_layer_size':32, 'epsilon_decay':500}
    agent = DQN_agent.DQN_agent(s, a, hp, double_dqn=True, dueling_dqn=False, prioritized_memory=False, noisy_MLP=False)
    return env_name, env, agent, score_scope, solved_score

def solve_lunar_lander():
    env_name="LunarLander-v2"; s=8; a=4; score_scope=100; solved_score=200
    env = gym.make(env_name)
    hp = {'lr':0.001, "min_playback":1000, "max_playback":1000000, "update_freq": 500, 'hiden_layer_size':256, 'epsilon_decay':10000}
    agent = DQN_agent.DQN_agent(s, a, hp, double_dqn=True, dueling_dqn=False, prioritized_memory=False, noisy_MLP=True)
    # agent = DiscretePPO.PPO_descrete_action(s, a)
    return env_name, env, agent, score_scope, solved_score


def solve_breakout():
    env_name="BreakoutNoFrameskip-v4"
    s=(4,84,84)
    a=4
    score_scope=100
    solved_score=20
    hp = {'lr':0.00001, "min_playback":50000, "max_playback":1000000, "update_freq": 10000, 'learn_freq':4, "normalize_state":True, 'epsilon_decay':1000000}
    agent = DQN_agent.DQN_agent(s, a, hp, double_dqn=True, dueling_dqn=False, prioritized_memory=False, noisy_MLP=False)
    env = get_final_env(env_name, frame_stack=True)

    return env_name, env, agent, score_scope, solved_score

def solve_pong():
    env_name="PongNoFrameskip-v4"
    env = get_final_env(env_name, frame_stack=False)
    s=(1,84,84)
    a=4
    score_scope=100
    solved_score=20
    hp = {'lr':0.0001, "min_playback":1000, "max_playback":100000, "update_freq": 1000, 'hiden_layer_size':512, "normalize_state":True, 'epsilon_decay':30000}
    agent = DQN_agent.DQN_agent(s, a, hp , double_dqn=True, dueling_dqn=False, prioritized_memory=False, noisy_MLP=False)
    return env_name, env, agent, score_scope, solved_score

if  __name__ == '__main__':
    SEED=2
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # env_name, env, agent, score_scope, solved_score = solve_cart_pole()
    env_name, env, agent, score_scope, solved_score = solve_pendulum()
    # env_name, env, agent, score_scope, solved_score = solve_lunar_lander()
    # env_name, env, agent, score_scope, solved_score = solve_bipedal_walker()
    # env_name, env, agent, score_scope, solved_score = solve_pong()
    # env_name, env, agent, score_scope, solved_score = solve_breakout()

    # Train
    os.makedirs("Training", exist_ok=True)
    TRAIN_DIR = os.path.join("Training", env_name)
    os.makedirs(TRAIN_DIR, exist_ok=True)

    train(env, agent, score_scope, solved_score)

    # # Test
    # render=False
    # if render:
    #     from pyglet.gl import *  # Fixes rendering issues of openAi gym with wrappers
    # score = test(env, agent, 1, render=True)
    # print("Reward over %d episodes: %f"%(3, score))
