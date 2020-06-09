import numpy as np
import os
from time import time, sleep
import random
import gym
from discrete_agents import *
from continous_agents import *
from hybrid_agents import *
import train_logger
import torch
from ExternalAtariWrappers import get_final_env

MAX_TRAIN_EPISODES = 1000000
TEST_EPISODES=1
CHECKPOINT_STEPS=0.2

def run_episode(env, agent):
    episode_rewards = []
    done = False
    state = env.reset()
    # lives = env.unwrapped.ale.lives()
    while not done:
        action = agent.process_new_state(state)
        state, reward, done, info = env.step(action)
        is_terminal = done
        # cur_life = env.unwrapped.ale.lives()
        # if cur_life < lives:
        #     is_terminal = True
        #     lives = cur_life
        if hasattr(env, '_max_episode_steps'):
            is_terminal = done and len(episode_rewards) < env._max_episode_steps
        agent.process_output(state, reward, is_terminal)
        episode_rewards += [reward]

    return episode_rewards


def train(env, agent, score_scope, solved_score, log_frequency=1, test_frequency=1000):
    next_progress_checkpoint = 1
    next_test_progress_checkpoint = 1

    # Define loggers
    logger = train_logger.plt_logger(score_scope, log_frequency,  os.path.join(TRAIN_DIR,  agent.name))
    # logger = train_logger.logger(score_scope, log_frequency)
    # logger.log_test(test(env, agent, TEST_EPISODES))
    for i in range(MAX_TRAIN_EPISODES):

        episode_rewards = run_episode(env, agent)

        if (i+1) % test_frequency == 0:
            last_test_score = test(env, agent, TEST_EPISODES)
            logger.log_test(last_test_score)
            if last_test_score >= next_test_progress_checkpoint * CHECKPOINT_STEPS * solved_score:
                agent.save_state(os.path.join(TRAIN_DIR, agent.name + "_test_%.5f_weights.pt" % last_test_score))
                next_test_progress_checkpoint += 1

        logger.update_train_episode(episode_rewards)
        if (i+1) % log_frequency == 0:
            last_k_scores = logger.output_stats(agent.get_stats())
            if last_k_scores >= next_progress_checkpoint*CHECKPOINT_STEPS*solved_score:
                agent.save_state(os.path.join(TRAIN_DIR, agent.name, agent.name + "_%.5f_weights.pt"%last_k_scores))
                next_progress_checkpoint += 1

            if last_k_scores > solved_score:
                print("Solved in %d episodes"%i)
                break

    env.close()


def test(env,  actor, test_episodes=1, render=False, delay=0.0):
    actor.train = False
    episodes_total_rewards = []
    for i in range(test_episodes):
        done = False
        state = env.reset()
        all_rewards = []
        while not done:
            if render:
                env.render()
                sleep(delay)
            action = actor.process_new_state(state)
            state, reward, done, info = env.step(action)
            all_rewards += [reward]

        episodes_total_rewards += [np.sum(all_rewards)]
    score = np.mean(episodes_total_rewards)
    env.close()
    actor.train=True
    return score


def solve_cart_pole():
    env_name="CartPole-v1"; s=4; a=2;score_scope=100; solved_score=195
    env = gym.make(env_name)
    ### With DQN
    # hp = {'lr':0.001, "min_playback":0, "max_playback":1000000, "update_freq": 100, 'hiden_layer_size':32, 'epsilon_decay':500}
    # agent = DQN_agent.DQN_agent(s, a, hp, double_dqn=True, dueling_dqn=False, prioritized_memory=False, noisy_MLP=False)


    # # With Vanila Actor-Critic
    # hp = {'lr':0.003, 'batch_size':150}
    # agent = GenericActorCritic.ActorCritic(s,a,hp)

    # With PPO
    hp = {'lr':0.001, 'epoch_size':400, 'epochs':4}
    agent = PPO.HybridPPO(s, a, hp)


    return env_name, env, agent, score_scope, solved_score


def solve_pendulum():
    env_name="Pendulum-v0";s=3; score_scope=100; solved_score=-200
    env = gym.make(env_name)
    a = [env.action_space.low, env.action_space.high]


    # # With Vanila Actor-Critic
    # hp = {'lr':0.0005, 'batch_size':1000}
    # agent = GenericActorCritic.ActorCritic(s, a,hp)

    # # With PPO
    # hp = {'lr':0.01, 'epoch_size':1000, 'epochs':4}
    # agent = PPO.HybridPPO(s, a, hp)

    # With TD3
    hp = {'actor_lr':0.00025, 'critic_lr':0.0002, "exploration_steps":5000, "min_memory_for_learning":10000, "batch_size": 256}
    agent = TD3.TD3(s, env.action_space, a, hp, train=True)


    return env_name, env, agent, score_scope, solved_score


def solve_lunar_lander():
    env_name="LunarLander-v2"; s=8; a=4; score_scope=100; solved_score=200
    env = gym.make(env_name)
    hp = {'lr':0.001, "min_playback":1000, "max_playback":1000000, "update_freq": 500, 'hiden_layer_size':256, 'epsilon_decay':10000}
    agent = DQN_agent.DQN_agent(s, a, hp, double_dqn=True, dueling_dqn=False, prioritized_memory=False, noisy_MLP=True)
    # agent = DiscretePPO.PPO_descrete_action(s, a)
    return env_name, env, agent, score_scope, solved_score

def solve_bipedal_walker():
    env_name="BipedalWalker-v3"; s=24; score_scope=100; solved_score=500
    env = gym.make(env_name)
    hp = {'actor_lr':0.00025, 'critic_lr':0.00025}#, "exploration_steps":5000, "min_memory_for_learning":10000, "batch_size": 256}
    agent = TD3.TD3(s, env.action_space, [env.action_space.low, env.action_space.high], hp, train=True)
    # agent.load_state('Trained_models/BipedalWalker-v3/TD3_lr[0.0003]_b[256]_tau[0.0050]_uf[2]/TD3_lr[0.0003]_b[256]_tau[0.0050]_uf[2]_test_309.12538_weights.pt')
    return env_name, env, agent, score_scope, solved_score

def solve_pong():
    env_name="PongNoFrameskip-v4"
    env = get_final_env(env_name, frame_stack=False)
    s=(1,84,84)
    a=4
    score_scope=100
    solved_score=20
    hp = {'lr':0.0001, "min_playback":1000, "max_playback":100000, "update_freq": 1000, 'hiden_layer_size':512, "normalize_state":True, 'epsilon_decay':30000}
    agent = DQN_agent.DQN_agent(s, a, hp , double_dqn=True, dueling_dqn=True, prioritized_memory=False, noisy_MLP=False)
    # agent.load_state('Trained_models/PongNoFrameskip-v4/DobuleDQN-DuelingDqn-Dqn-lr[0.00008]_b[32]_lf[1]_uf[1000]/DobuleDQN-DuelingDqn-Dqn-lr[0.00008]_b[32]_lf[1]_uf[1000]_test_21.00000_weights.pt')
    return env_name, env, agent, score_scope, solved_score


def solve_breakout():
    env_name="BreakoutNoFrameskip-v4"
    env_name="BreakoutDeterministic-v4"
    s=(4,84,84)
    a=4
    score_scope=100
    solved_score=20
    hp = {'lr':0.00001, "min_playback":50000, "max_playback":1000000, "update_freq": 10000, 'learn_freq':4, "normalize_state":True, 'epsilon_decay':5000000}
    hp = {'lr':0.00001, "min_playback":32, "max_playback":1000000, "update_freq": 10000, 'learn_freq':4, "normalize_state":True, 'epsilon_decay':5000000}
    agent = DQN_agent.DQN_agent(s, a, hp, double_dqn=True, dueling_dqn=False, prioritized_memory=False, noisy_MLP=False)
    env = get_final_env(env_name, frame_stack=True, episode_life=False)

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

    # Test
    # render=True
    # if render:
    #     from pyglet.gl import *  # Fixes rendering issues of openAi gym with wrappers
    # score = test(env, agent, 1, render=render, delay=0.025)
    # print("Reward over %d episodes: %f"%(3, score))
