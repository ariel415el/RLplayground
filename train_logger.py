from _collections import deque
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
from time import time
from tensorboardX import SummaryWriter
import torch

class train_stats(object):
    def __init__(self, name, x, y):
        self.name = name
        self.xs = [x]
        self.ys = [y]

    def add(self, x, y):
        self.xs += [x]
        self.ys += [y]

    def plot(self, path):
        plt.plot(self.xs, self.ys, label=self.name)
        plt.legend()
        plt.savefig(path)
        plt.clf()


class logger(object):
    def __init__(self, k, log_frequency=10):
        self.log_frequency = log_frequency
        self.last_episodes_total_rewards = deque(maxlen=k)
        self.train_start = time()
        self.total_steps = 0
        self.done_episodes = 0
        self.last_time = self.train_start
        self.last_steps = 0
        self.agent_train_stats = {}

    def get_last_k_episodes_mean(self):
        return np.mean(self.last_episodes_total_rewards)

    def update_agent_stats(self, name, x, y):
        if name in self.agent_train_stats:
            self.agent_train_stats[name].add(x,y)
        else:
            self.agent_train_stats[name] = train_stats(name, x, y)

    def update_train_episode(self, episode_rewards):
        self.last_episodes_total_rewards.append(np.sum(episode_rewards))
        self.total_steps += len(episode_rewards)
        self.done_episodes += 1
        if self.done_episodes % self.log_frequency == 0 :
            self.output_stats()

    def output_stats(self):
        time_passed = time() - self.train_start
        print('Episodes done: ', self.done_episodes)
        print("\t# Steps %d, time %d mins; avg-%d %.2f:" % (self.total_steps, time_passed / 60, len(self.last_episodes_total_rewards), self.get_last_k_episodes_mean()))
        print("\t# steps/sec avg: %.3f " % (self.total_steps / time_passed))
        # print("\t# steps/sec avg: %.3f " % ((self.total_steps- self.last_steps) / (time_passed - self.last_time)))
        print("\t# Agent stats: ", ";".join([name+" : %.5f"%self.agent_train_stats[name].ys[-1] for name in self.agent_train_stats]))
        self.last_steps = self.total_steps
        self.last_time = time_passed

    def log_test(self, score):
        print("Test score: %.3f "%score)

class TB_logger(logger):
    def __init__(self, k, log_frequency, logdir):
        super(TB_logger, self).__init__(k, log_frequency)
        self.log_frequency = log_frequency
        self.k = k
        self.tb_writer = SummaryWriter(os.path.join(logdir,'tensorboard'))

    def update_agent_stats(self, name, x, y):
        self.tb_writer.add_scalar(name, torch.tensor(y), global_step=x)

    def update_train_episode(self, episode_rewards):
        self.done_episodes += 1
        if self.done_episodes % self.log_frequency ==0:
            self.output_stats()
        self.total_steps += len(episode_rewards)
        episode_total_score = np.sum(episode_rewards)
        self.last_episodes_total_rewards.append(episode_total_score)
        self.tb_writer.add_scalars('Episode_score',{"Score": torch.tensor(episode_total_score),
                                                   'Last-%d episode'%self.k: torch.tensor(self.get_last_k_episodes_mean())},
                                  global_step=self.done_episodes)

        self.tb_writer.add_scalar('Episode-Length', torch.tensor(len(episode_rewards)),
                                  global_step=self.done_episodes)

class plt_logger(logger):
    def __init__(self, k, log_frequency, logdir):
        super(plt_logger, self).__init__(k, log_frequency)
        os.makedirs(logdir, exist_ok=True)
        self.k = k
        self.logdir=logdir
        self.all_episode_lengths = []
        self.all_episode_total_scores = []
        self.all_avg_last_k = []
        self.test_scores = []

    def update_train_episode(self, episode_rewards):
        self.all_episode_lengths += [len(episode_rewards)]
        self.all_episode_total_scores += [np.sum(episode_rewards)]
        super(plt_logger, self).update_train_episode(episode_rewards)
        self.all_avg_last_k += [self.get_last_k_episodes_mean()]

    def output_stats(self, actor_stats=None, by_step=False):
        super(plt_logger, self).output_stats()
        xs = np.arange(1, len(self.all_episode_lengths) + 1)
        x_label = 'Episode #'
        if by_step:
            xs = np.cumsum(self.all_episode_lengths)
            x_label = 'Step #'
        plt.plot(xs, self.all_episode_lengths)
        plt.ylabel('Length')
        plt.xlabel(x_label)
        plt.savefig(os.path.join(self.logdir, "Episode-lengths.png"))
        plt.clf()

        plt.plot(xs, self.all_episode_total_scores, label='Episode-score')
        plt.plot(xs, self.all_avg_last_k, label='Score-last %d' % self.k)
        plt.ylabel('Score')
        plt.xlabel(x_label)
        plt.legend()
        plt.savefig(os.path.join(self.logdir, "Episode-scores.png"))
        plt.clf()

        for train_stats in self.agent_train_stats:
            self.agent_train_stats[train_stats].plot(os.path.join(self.logdir, train_stats+".png"))


    def log_test(self, score):
        super(plt_logger, self).log_test(score)
        self.test_scores += [score]
        plt.plot( np.linspace(start=1, stop=max(2,len(self.all_avg_last_k)), num=len(self.test_scores)), self.test_scores)
        plt.ylabel('test_score-last %d'%self.k)
        plt.xlabel('Episode #')
        plt.savefig(os.path.join(self.logdir, "Test_scores.png"))
        plt.clf()