import numpy as np
import os
from time import time, sleep
import random
import gym
from discrete_agents import *
from continous_agents import *
import train_logger
import torch
from utils import measure_time, image_preprocess_wrapper, my_image_level_wrapper
from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.atari_preprocessing import AtariPreprocessing

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


if  __name__ == '__main__':
    kwargs = {}
    # Choose enviroment
    # ENV_NAME="CartPole-v1"; s=4; a=2;score_scope=100; solved_score=195
    ENV_NAME="Breakout-v0"; s=(105,80, 4); a=3;score_scope=100; solved_score=195;kwargs = {'frameskip':0}
    # ENV_NAME='PixelChopper';s=7;a=2;score_scope=100; solved_score=100
    # ENV_NAME="LunarLander-v2"; s=(105,80); a=4; score_scope=20; solved_score=200
    # ENV_NAME="LunarLander-v2"; s=8; a=4; score_scope=20; solved_score=200
    # ENV_NAME="LunarLanderContinuous-v2";s=8; score_scope=100; solved_score=200
    # ENV_NAME="Pendulum-v0";s=3; score_scope=100; solved_score=-200
    # ENV_NAME="BipedalWalker-v3"; s=24; score_scope=100; solved_score=500
    # ENV_NAME="BipedalWalkerHardcore-v3"; s=24; score_scope=100; solved_score=300

    env = gym.make(ENV_NAME, **kwargs)
    # env = my_image_level_wrapper(env)
    env = image_preprocess_wrapper(env)
    # env = PLE2GYM_wrapper(render=False)

    # set seeds
    SEED=0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    env.seed(SEED)

    # Create agent
    NUM_EPISODES = 10000
    actor = DQN_agent.DQN_agent(s, a, train=True)
    # actor = DiscretePPO.PPO_descrete_action(s, a, NUM_EPISODES, train=True)
    # actor = vanila_policy_gradient_agent(s, a, NUM_EPISODES, train=True)
    # actor = actor_critic_agent(s, a, NUM_EPISODES, train=True, critic_objective="Monte-Carlo")
    # actor = actor_critic_agent(s, bounderies, NUM_EPISODES, train=True, critic_objective="Monte-Carlo")
    # actor = DDPG.DDPG(s, bounderies, NUM_EPISODES, train=True)
    # actor = TD3.TD3(s, [env.action_space.low, env.action_space.high], NUM_EPISODES, train=True, action_space=env.action_space)
    # actor = PPO.PPO(s, bounderies, NUM_EPISODES, train=True)

    # env = PixelObservationWrapper(env)

    # Train
    os.makedirs("Training", exist_ok=True)
    TRAIN_DIR = os.path.join("Training", ENV_NAME)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    # trained_weights = None
    # trained_weights = os.path.join(TRAIN_DIR, actor.name + "_trained_weights.pt")
    # trained_weights =  '/projects/RL/RL_implementations/Training/LunarLander-v2/PPO_lr[0.0010]_b[3000]/PPO_lr[0.0010]_b[3000]_124.67556_weights.pt'
    # actor.load_state(trained_weights)

    # actor.hyper_parameters['exploration_steps'] = -1
    # actor.hyper_parameters['min_memory_for_learning'] = -1
    # actor.hyper_parameters['actor_lr'] = 0.005
    # actor.hyper_parameters['critic_lr'] = 0.005
    train(env, actor, NUM_EPISODES, score_scope, solved_score)

    # Test
    # actor.train = False
    # score = test(env, actor, 3, render=True)
    # print("Reward over %d episodes: %f"%(3, score))
