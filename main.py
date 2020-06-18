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
import train

def solve_cart_pole():
    env_name="CartPole-v1"; s=4; a=2;score_scope=100; solved_score=195
    env = gym.make(env_name)
    # ## With DQN
    # hp = {'lr':0.001, "min_playback":0, "max_playback":1000000, "update_freq": 100, 'hiden_layer_size':32, 'epsilon_decay':500}
    # agent = DQN_agent.DQN_agent(s, a, hp, double_dqn=True, dueling_dqn=False, prioritized_memory=False, noisy_MLP=False)

    # With VanilaPG
    hp = {'lr':0.001, 'batch_episodes':1}
    agent = VanilaPolicyGradient.VanilaPolicyGradient(s, a, hp)

    # # With Actor-Critic
    # hp = {'lr':0.001, 'batch_episodes':1, 'GAE': 0.9}
    # agent = GenericActorCritic.ActorCritic(s,a,hp)

    # # With PPO
    # hp = {'lr':0.001, 'batch_episodes':1, 'epochs':4, 'GAE':1.0, 'value_clip':0.3, 'grad_clip':None}
    # agent = PPO.HybridPPO(s, a, hp)


    return env_name, env, agent, score_scope, solved_score


def solve_mountain_car():
    env_name="MountainCar-v0"; s=2; a=3;score_scope=100; solved_score=195
    env = gym.make(env_name)
    # ## With DQN
    # hp = {'lr':0.001, "min_playback":0, "max_playback":1000000, "update_freq": 100, 'hiden_layer_size':32, 'epsilon_decay':500}
    # agent = DQN_agent.DQN_agent(s, a, hp, double_dqn=True, dueling_dqn=False, prioritized_memory=False, noisy_MLP=False)

    # With VanilaPG
    hp = {'lr':0.001, 'batch_episodes':8}
    agent = VanilaPolicyGradient.VanilaPolicyGradient(s, a, hp)

    # # With Actor-Critic
    # hp = {'lr':0.001, 'batch_episodes':1, 'GAE': 0.9}
    # agent = GenericActorCritic.ActorCritic(s,a,hp)

    # # With PPO
    # hp = {'lr':0.001, 'batch_episodes':4, 'epochs':8, 'GAE':0.95, 'value_clip':None, 'grad_clip':None, 'hidden_layers':[128,128]}
    # # agent = PPO.HybridPPO(s, a, hp)
    # agent = PPO_ICM.HybridPPO_ICM(s, a, hp)


    return env_name, env, agent, score_scope, solved_score


def solve_pendulum():
    env_name="Pendulum-v0";s=3; score_scope=100; solved_score=-200
    env = gym.make(env_name)
    a = [env.action_space.low, env.action_space.high]

    # With VanilaPG
    hp = {'lr':0.0001, 'batch_episodes':32, 'hidden_layers':[400,300]}
    agent = VanilaPolicyGradient.VanilaPolicyGradient(s, a, hp)

    # # With Vanila Actor-Critic
    # hp = {'lr':0.0001, 'batch_episodes':64, 'GAE':0.95, 'hidden_layers':[400,400]}
    # agent = GenericActorCritic.ActorCritic(s, a,hp)

    # # With PPO
    # hp = {'lr':0.001, 'batch_episodes':45, 'epochs':10, 'GAE':0.95, 'epsiolon_clip': 0.1, 'value_clip':0.1, 'grad_clip':None, 'entropy_weight':0.01, 'hidden_layer_size':512}
    # agent = PPO.HybridPPO(s, a, hp)

    # # DDPG
    # hp ={'actor_lr':0.0001, 'critic_lr':0.001, 'batch_size':64, 'min_playback':1000, 'layer_dims':[400,300], 'tau':0.001, "update_freq":1, 'learn_freq':1}
    # agent = DDPG.DDPG(s, a, hp)

    # # With TD3
    # hp = {'actor_lr':0.0003, 'critic_lr':0.00025, "exploration_steps":5000, "min_memory_for_learning":10000, "batch_size": 128}
    # hp = {'actor_lr':0.0005, 'critic_lr':0.0005, "exploration_steps":1000, "min_memory_for_learning":5000, "batch_size": 64}
    # agent = TD3.TD3(s, env.action_space, a, hp, train=True)

    return env_name, env, agent, score_scope, solved_score


def solve_lunar_lander():
    env_name="LunarLander-v2"; s=8; a=4; score_scope=100; solved_score=200
    env = gym.make(env_name)
    # # With DQN
    # hp = {'lr':0.0007, "min_playback":1000, "max_playback":1000000, "update_freq": 500, 'hiden_layer_size':256, 'epsilon_decay':10000}
    # agent = DQN_agent.DQN_agent(s, a, hp, double_dqn=True, dueling_dqn=False, prioritized_memory=False, noisy_MLP=False)

    # # # With VanilaPG
    # hp = {'lr':0.001, 'batch_episodes':32, 'hidden_layers':[64,64,128]}
    # agent = VanilaPolicyGradient.VanilaPolicyGradient(s, a, hp)

    # # With Actor-Critic
    # hp = {'lr':0.005, 'batch_episodes':8, 'GAE': 0.96, 'hidden_layer_size':16}
    # agent = GenericActorCritic.ActorCritic(s,a,hp)

    # With PPO
    # hp = {'lr':0.00025, 'batch_episodes':8, 'epochs':3, 'GAE':0.95, 'epsiolon_clip': 0.1, 'value_clip':None, 'grad_clip':0.5, 'entropy_weight':0.01, 'hidden_layer_size':64}
    hp = {'lr':0.0005, 'batch_episodes':32, 'epochs':10, 'GAE':0.95, 'epsiolon_clip': 0.5, 'value_clip':None, 'grad_clip':None, 'entropy_weight':0.01, 'hidden_layer_size':[64,64,128]}
    # agent = PPO.HybridPPO(s, a, hp)
    agent = PPO_ICM.HybridPPO_ICM(s, a, hp)

    return env_name, env, agent, score_scope, solved_score


def solve_continous_lunar_lander():
    env_name="LunarLanderContinuous-v2"; s=8; score_scope=100; solved_score=200
    env = gym.make(env_name)
    a = [env.action_space.low, env.action_space.high]

    # # With PPO
    # hp = {'lr':0.001, 'batch_episodes':16, 'epochs':32, 'GAE':1.0, 'epsiolon_clip': 0.2, 'value_clip':None, 'grad_clip':None, 'entropy_weight':0.01, 'hidden_layer_size':256}
    # agent = PPO.HybridPPO(s, a, hp)

    # # DDPG
    # hp ={'actor_lr':0.00005, 'critic_lr':0.0005, 'batch_size':64, 'min_playback':0, 'layer_dims':[400,200], 'tau':0.001, "update_freq":1, 'learn_freq':1}
    # agent = DDPG.DDPG(s, a, hp)

    # DDPG
    hp ={'actor_lr':0.0001, 'critic_lr':0.001, 'batch_size':100, 'min_playback':0, 'layer_dims':[400,200], 'tau':0.001, "update_freq":1, 'learn_freq':1}
    agent = DDPG.DDPG(s, a, hp)

    # # With TD3
    # hp = {'actor_lr':0.0003, 'critic_lr':0.00025, "exploration_steps":5000, "min_memory_for_learning":10000, "batch_size": 128}
    # agent = TD3.TD3(s, env.action_space, a, hp, train=True)

    return env_name, env, agent, score_scope, solved_score


def solve_bipedal_walker():
    env_name="BipedalWalker-v3"; s=24; score_scope=100; solved_score=500
    env = gym.make(env_name)
    a = [env.action_space.low, env.action_space.high]

    # With PPO
    hp = {'lr':0.00025, 'batch_episodes':16, 'epochs':32, 'GAE':1.0, 'epsiolon_clip': 0.2, 'value_clip':None, 'grad_clip':None, 'entropy_weight':0.01, 'hidden_layer_size':256}
    agent = PPO.HybridPPO(s, a, hp)

    # DDPG
    hp ={'actor_lr':0.0001, 'critic_lr':0.001, 'batch_size':100, 'min_playback':0, 'layer_dims':[400,200], 'tau':0.001, "update_freq":1, 'learn_freq':1}
    agent = DDPG.DDPG(s, a, hp)


    # # With TD3
    # hp = {'actor_lr':0.00025, 'critic_lr':0.00025}#, "exploration_steps":5000, "min_memory_for_learning":10000, "batch_size": 256}
    # agent = TD3.TD3(s, env.action_space, a, hp, train=True)

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
    # env_name="BreakoutDeterministic-v4"
    s=(4,84,84)
    a=4
    score_scope=100
    solved_score=20
    hp = {'lr':0.00001, "min_playback":50000, "max_playback":1000000, "update_freq": 10000, 'learn_freq':4, "normalize_state":True, 'epsilon_decay':5000000}
    hp = {'lr':0.00001, "min_playback":32, "max_playback":1000000, "update_freq": 10000, 'learn_freq':4, "normalize_state":True, 'epsilon_decay':5000000}
    agent = DQN_agent.DQN_agent(s, a, hp, double_dqn=True, dueling_dqn=False, prioritized_memory=False, noisy_MLP=False)
    env = get_final_env(env_name, frame_stack=True, episode_life=False)

    return env_name, env, agent, score_scope, solved_score


def solve_seaquest():
    env_name="SeaquestNoFrameskip-v4"
    s=(1,84,84)
    env = get_final_env(env_name, frame_stack=False, episode_life=True)
    a=4
    score_scope=100
    solved_score=20

    # With DQN
    # hp = {'lr':0.00001, "min_playback":50000, "max_playback":1000000, "update_freq": 10000, 'learn_freq':4, "normalize_state":True, 'epsilon_decay':5000000}
    # agent = DQN_agent.DQN_agent(s, a, hp, double_dqn=True, dueling_dqn=False, prioritized_memory=False, noisy_MLP=False)

    # With actor-critic
    hp = {'lr':0.0007, 'batch_size':4000}
    agent = GenericActorCritic.ActorCritic(s, a, hp)

    return env_name, env, agent, score_scope, solved_score


if  __name__ == '__main__':
    SEED=0
    TRAIN_ROOT="TEST_TRAINING"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env_name, env, agent, score_scope, solved_score = solve_cart_pole()
    # env_name, env, agent, score_scope, solved_score = solve_mountain_car()
    # env_name, env, agent, score_scope, solved_score = solve_pendulum()
    # env_name, env, agent, score_scope, solved_score = solve_lunar_lander()
    # env_name, env, agent, score_scope, solved_score = solve_continous_lunar_lander()
    # env_name, env, agent, score_scope, solved_score = solve_bipedal_walker()
    # env_name, env, agent, score_scope, solved_score = solve_pong()
    # env_name, env, agent, score_scope, solved_score = solve_breakout()
    # env_name, env, agent, score_scope, solved_score = solve_seaquest()
    env.seed(SEED)

    # Train
    # os.makedirs(TRAIN_ROOT, exist_ok=True)
    # TRAIN_DIR = os.path.join(TRAIN_ROOT, env_name)
    # os.makedirs(TRAIN_DIR, exist_ok=True)
    train_dir =  os.path.join(TRAIN_ROOT, env_name,  agent.name)
    os.makedirs(train_dir, exist_ok=True)
    assert(os.path.exists(train_dir))

    # logger = train_logger.plt_logger(k=score_scope, log_frequency=10, logdir=train_dir)
    logger = train_logger.TB_logger(k=score_scope, log_frequency=10, logdir=train_dir)
    # logger = train_logger.logger(score_scope, log_frequency=10)

    agent.set_reporter(logger)
    train.train_agent(env, agent, solved_score, train_dir, logger, test_frequency=100, train_episodes=10000, test_episodes=2, save_videos=True, checkpoint_steps=0.2)

    # # Test
    # agent.load_state('/projects/RL/RL_implementations/Trained_models/LunarLanderContinuous-v2/DDPG_[400, 200]_BN_lr[0.0001]_b[64]_tau[0.0050]_uf[1]_lf[1]/DDPG_[400, 200]_BN_lr[0.0001]_b[64]_tau[0.0050]_uf[1]_lf[1]_204.60898_weights.pt')
    # render=True
    # if render:
    #     from pyglet.gl import *  # Fixes rendering issues of openAi gym with wrappers
    # score = train.test(env, agent, 3, render=render, delay=0.025)
    # print("Reward over %d episodes: %f"%(3, score))
