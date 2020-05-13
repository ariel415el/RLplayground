from _collections import deque
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import matplotlib.pyplot as plt

class logger(object):
    def __init__(self, k):
        self.total_rewards = deque(maxlen=k)

    def log(self, episode_number, episode_rewards, num_steps, time_passed, actor_stats):
        self.total_rewards.append(np.sum(episode_rewards))
        last_k_scores = np.mean(self.total_rewards)
        print('Episode: ', episode_number)
        print("\t# Step %d, time %d mins; avg-%d %.2f:" % (num_steps, time_passed / 60, len(self.total_rewards),  last_k_scores))
        print("\t# steps/sec", num_steps / time_passed)
        print("\t# Agent stats: ", actor_stats)

class TB_logger(logger):
    def __init__(self, k, tb_writer):
        super(TB_logger, self).__init__(k)
        self.tb_writer = tb_writer

    def log(self, episode_number, episode_rewards, num_steps, time_passed, actor_stats):
        super(TB_logger, self).log(episode_number, episode_rewards, num_steps, time_passed, actor_stats)
        last_k_scores = np.mean(self.total_rewards)
        self.tb_writer.add_scalar('1.last_%s_episodes_avg'%len(self.total_rewards), torch.tensor(last_k_scores), global_step=episode_number)
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