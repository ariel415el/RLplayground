import gym
from descrete_agents.DQN_agent import *
from descrete_agents.vanila_policy_gradient import *
from descrete_agents.actor_critic import *
from continous_agents.actor_critic import actor_critic_agent
from continous_agents.DDPG import DDPG_agent
from continous_agents.PPO import PPO
from continous_agents.TD3 import TD3
import numpy as np
from time import time, sleep
from torch.utils.tensorboard import SummaryWriter
from _collections import deque
import random
from matplotlib import pyplot as plt
from gym import wrappers

class logger(object):
    def __init__(self, k):
        self.total_rewards = deque(maxlen=k)

    def log(self, episode_number, episode_rewards, num_steps, time_passed, actor_stats):
        self.total_rewards.append(np.sum(episode_rewards))
        last_k_scores = np.mean(self.total_rewards)
        print('Episode: ', episode_number)
        print("\t# Step %d, time %d mins; avg-100 %.2f:" % (num_steps, time_passed / 60, last_k_scores))
        print("\t# steps/sec", num_steps / time_passed)
        print("\t# Agent stats: ", actor.get_stats())

class TB_logger(logger):
    def __init__(self, k, tb_writer):
        super(TB_logger, self).__init__(k)
        self.tb_writer = tb_writer

    def log(self, episode_number, episode_rewards, num_steps, time_passed, actor_stats):
        super(TB_logger, self).log(episode_number, episode_rewards, num_steps, time_passed, actor_stats)
        last_k_scores = np.mean(self.total_rewards)
        self.tb_writer.add_scalar('1.last_100_episodes_avg', torch.tensor(last_k_scores), global_step=episode_number)
        self.tb_writer.add_scalar('2.episode_score', torch.tensor(np.sum(episode_rewards)), global_step=episode_number)
        self.tb_writer.add_scalar('3.episode_length', len(episode_rewards), global_step=episode_number)
        self.tb_writer.add_scalar('4.avg_rewards', torch.tensor(np.mean(episode_rewards)), global_step=episode_number)
        # cur_time = max(1, int(time() - train_start))
        # self.tb_writer.add_scalar('5.episode_score_time_scaled', torch.tensor(episode_score), global_step=cur_time)

class plt_logger(logger):
    def __init__(self, k, logdir):
        super(plt_logger, self).__init__(k)
        os.makedirs(logdir, exist_ok=True)
        self.k = k
        self.logdir=logdir
        self.all_episode_lengths = []
        self.all_episode_total_scores = []
        self.all_avg_last_k = []

    def log(self, episode_number, episode_rewards, num_steps, time_passed, actor_stats):
        super(plt_logger, self).log(episode_number, episode_rewards, num_steps, time_passed, actor_stats)
        last_k_scores = np.mean(self.total_rewards)
        self.all_episode_lengths += [len(episode_rewards)]
        self.all_episode_total_scores += [np.sum(episode_rewards)]
        self.all_avg_last_k += [last_k_scores]

        plt.plot(np.arange(1, len(self.all_episode_lengths) + 1), self.all_episode_lengths)
        plt.ylabel('Length')
        plt.xlabel('Episode #')
        plt.savefig(os.path.join(self.logdir,"Episode-lengths.png"))
        plt.clf()

        plt.plot(np.arange(1, len(self.all_episode_total_scores) + 1), self.all_episode_total_scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig(os.path.join(self.logdir,"Episode-scores.png"))
        plt.clf()

        plt.plot(np.arange(1, len(self.all_avg_last_k) + 1), self.all_avg_last_k)
        plt.ylabel('Score-last %d'%self.k)
        plt.xlabel('Episode #')
        plt.savefig(os.path.join(self.logdir,"Episode-avg-last-%d.png"%self.k))
        plt.clf()

        return last_k_scores

def train(env, actor, train_episodes, score_scope):
    train_start = time()
    # logger = TB_logger(200, SummaryWriter(log_dir=os.path.join(TRAIN_DIR, "tensorboard_outputs",  actor.name)))
    logger = plt_logger(score_scope, os.path.join(TRAIN_DIR,  actor.name))
    num_steps = 0
    for i in range(train_episodes):

        done = False
        state = env.reset()
        episode_rewards = []
        while not done:

            action = actor.process_new_state(state)
            state, reward, done, info = env.step(action)
            actor.process_output(state, reward, done)
            num_steps+=1
            episode_rewards += [reward]

        last_k_scores = logger.log(i, episode_rewards, num_steps, max(1, int(time() - train_start)), actor.get_stats())

        if last_k_scores > 200:
            print("Solved in %d episodes",%i)

    actor.save_state(os.path.join(TRAIN_DIR, actor.name + "_trained_weights.pt"))

    env.close()

def test(env,  actor):
    actor.load_state(os.path.join(TRAIN_DIR, actor.name + "_trained_weights.pt"))
    i = 0
    while True:
        i+=1
        done = False
        state = env.reset()
        all_rewards = []
        while not done:
            if i ==12:
                env.render()
            action = actor.process_new_state(state)
            state, reward, done, info = env.step(action)
            all_rewards += [reward]
        print("total reward: %f, # steps %d"%(np.sum(all_rewards),len(all_rewards)))
        env.close()

if  __name__ == '__main__':
    SEED=0
    random.seed(SEED)
    torch.manual_seed(SEED)
    # ENV_NAME="CartPole-v1"; s=4; a=2
    # ENV_NAME="LunarLander-v2"; s=8; a=4
    ENV_NAME="LunarLanderContinuous-v2";s=8;bounderies=[[-1,-1],[1,1]]; score_scope=20
    # ENV_NAME="Pendulum-v0";s=3;bounderies=[[-2],[2]]
    # ENV_NAME="BipedalWalker-v3"; s=24;bounderies=[[-1,-1,-1,-1],[1,1,1,1]]
    os.makedirs("Training", exist_ok=True)
    TRAIN_DIR = os.path.join("Training", ENV_NAME)
    os.makedirs(TRAIN_DIR, exist_ok=True)

    env = gym.make(ENV_NAME)
    env.seed(SEED)
    NUM_EPISODES = 10000
    # actor = DQN_agent(s, a, NUM_EPISODES, train=True)
    # actor = vanila_policy_gradient_agent(s, a, NUM_EPISODES, train=True)
    # actor = actor_critic_agent(s, a, NUM_EPISODES, train=True, critic_objective="Monte-Carlo")
    # actor = actor_critic_agent(s, bounderies, NUM_EPISODES, train=True, critic_objective="Monte-Carlo")
    # actor = DDPG_agent(s, bounderies, NUM_EPISODES, train=True)
    actor = TD3(s, bounderies, NUM_EPISODES, train=True)
    # actor = PPO(s, bounderies, NUM_EPISODES, train=True)

    train(env, actor, NUM_EPISODES, score_scope)
    # actor.train = False
    # actor.epsilon = 0.0
    # test(env, actor)