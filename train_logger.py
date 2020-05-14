from _collections import deque
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import matplotlib.pyplot as plt
from time import time

class logger(object):
    def __init__(self, k):
        self.last_episodes_total_rewards = deque(maxlen=k)
        self.total_steps  = 0
        self.train_start = time()
        self.last_time = 0
        self.test_scores = []

    def log_train_episode(self, episode_number, episode_rewards, actor_stats):
        time_passed = time() - self.train_start
        self.total_steps += len(episode_rewards)
        self.last_episodes_total_rewards.append(np.sum(episode_rewards))
        last_k_scores = np.mean(self.last_episodes_total_rewards)
        print('Episode: ', episode_number)
        print("\t# Step %d, time %d mins; reward: %.2f; avg-%d %.2f:" % (
            self.total_steps, time_passed / 60, self.last_episodes_total_rewards[-1], len(self.last_episodes_total_rewards), last_k_scores))
        print("\t# steps/sec now: %.3f avg: %.3f "%(
            len(episode_rewards) / (time_passed - self.last_time), self.total_steps /time_passed))
        print("\t# Agent stats: ", actor_stats)
        self.last_time = time_passed

        return last_k_scores

    def log_test(self, score):
        print("Test score: %.3f "%score)
        self.test_scores += [score]

class TB_logger(logger):
    def __init__(self, k, tb_writer):
        super(TB_logger, self).__init__(k)
        self.tb_writer = tb_writer

    def log_traine_episode(self, episode_number, episode_rewards, actor_stats):
        last_k_scores = super(TB_logger, self).log_train_episode(episode_number, episode_rewards, actor_stats)
        self.tb_writer.add_scalar('1.last_%s_episodes_avg'%len(self.last_episodes_total_rewards), torch.tensor(last_k_scores), global_step=episode_number)
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

    def log_train_episode(self, episode_number, episode_rewards, actor_stats):
        last_k_scores = super(plt_logger, self).log_train_episode(episode_number, episode_rewards, actor_stats)
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

    def log_test(self, score):
        super(plt_logger, self).log_test(score)
        plt.plot( np.linspace(start=1, stop=max(2,len(self.all_avg_last_k)), num=len(self.test_scores)), self.test_scores)
        plt.ylabel('test_score-last %d'%self.k)
        plt.xlabel('Episode #')
        plt.savefig(os.path.join(self.logdir, "Test_scores.png"))
        plt.clf()