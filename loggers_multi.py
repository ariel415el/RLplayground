from _collections import deque
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
from time import time
import torch
import pickle


class train_stats(object):
    def __init__(self, name, x, y):
        self.name = name
        self.xs = [x]
        self.ys = [y]

    def add(self, x, y):
        if x is None:
            x = len(self.ys)
        self.xs += [x]
        self.ys += [y]

    def plot(self, path, k=100):
        plt.plot(self.xs, self.ys, label=self.name, )
        avg = np.convolve(np.array(self.ys), np.ones(k)/float(k), 'valid')
        plt.plot(np.arange(len(avg)), avg, label='%d-avg'%k)
        plt.legend()
        plt.savefig(path)
        plt.clf()


def multi_env_logger(object):
    def __init__(self, k, log_frequency, logdir):
        self.k = k
        self.logdir = logdir
        self.all_episodes_total_rewards = []
        self.done_episodes = 0
        self.last_time = self.train_start
        self.agent_train_stats = {}

class logger(object):
    def __init__(self, k, log_frequency, logdir):
        self.k = k
        self.logdir = logdir
        self.log_frequency = log_frequency
        self.all_episodes_total_rewards = []
        self.train_start = time()
        self.total_steps = 0
        self.done_episodes = 0
        self.last_time = self.train_start
        self.agent_train_stats = {}
        self.agent_histograms = {}
        self.last_output_stats = 0

    def get_last_k_episodes_mean(self):
        return np.mean(self.all_episodes_total_rewards[-self.k:])

    def update_agent_stats(self, name, x, y):
        if name in self.agent_train_stats:
            self.agent_train_stats[name].add(x, y)
        else:
            self.agent_train_stats[name] = train_stats(name, x, y)

    def add_histogram(self, name, values):
        if name not in self.agent_histograms:
            self.agent_histograms[name] = list(values)
        else:
            self.agent_histograms[name] += list(values)

    def update_train_episodes(self, episodes_total_rewards, episodes_lengths):
        assert(len(episodes_lengths) == len(episodes_total_rewards))
        self.all_episodes_total_rewards += episodes_total_rewards
        self.total_steps += np.sum(episodes_lengths)
        self.done_episodes += len(episodes_total_rewards)
        if self.done_episodes - self.last_output_stats >= self.log_frequency:
            self.output_stats()

    def output_stats(self):
        self.last_output_stats = self.done_episodes
        time_passed = time() - self.train_start
        print('Episodes done: ', self.done_episodes)
        print("\t# Steps %d, time %d mins; avg-%d %.2f:" % (self.total_steps, time_passed / 60, self.k, self.get_last_k_episodes_mean()))
        print("\t# steps/sec avg: %.3f " % (self.total_steps / time_passed))
        print("\t# Agent stats:", "; ".join([name+":%.5f"%self.agent_train_stats[name].ys[-1] for name in self.agent_train_stats]))
        self.last_time = time_passed

    def log_test(self, score):
        print("Test score: %.3f "%score)

    def pickle_episode_scores(self):
        f = open(os.path.join(self.logdir, "episode_scores.pkl"), 'wb')
        pickle.dump(self.all_episodes_total_rewards, f)

class TB_logger(logger):
    def __init__(self, k, log_frequency, logdir):
        super(TB_logger, self).__init__(k, log_frequency, logdir)
        from tensorboardX import SummaryWriter
        self.tb_writer = SummaryWriter(os.path.join(logdir,'tensorboard'))

    def update_agent_stats(self, name, x, y):
        self.tb_writer.add_scalar(name, torch.tensor(y), global_step=x)

    def update_train_episodes(self, episodes_total_rewards, episodes_lengths):
        assert(len(episodes_lengths) == len(episodes_total_rewards))
        self.done_episodes += len(episodes_total_rewards)
        if self.done_episodes - self.last_output_stats >= self.log_frequency:
            self.output_stats()
        self.total_steps += sum(episodes_lengths)
        for score,length in zip(episodes_total_rewards, episodes_lengths):
            self.all_episodes_total_rewards += [score]
            self.tb_writer.add_scalars('Episode_score',{"Score": torch.tensor(score),
                                                        'Last-%d episode'%self.k: torch.tensor(self.get_last_k_episodes_mean())},
                                       global_step=self.done_episodes)

            self.tb_writer.add_scalar('Episode-Length', torch.tensor(len(length)),
                                      global_step=self.done_episodes)

class plt_logger(logger):
    def __init__(self, k, log_frequency, logdir):
        super(plt_logger, self).__init__(k, log_frequency, logdir)
        os.makedirs(logdir, exist_ok=True)
        self.all_episode_lengths = []
        self.all_avg_last_k = []
        self.test_scores = []

    def update_train_episodes(self, episodes_total_rewards, episodes_lengths):
        assert(len(episodes_lengths) == len(episodes_total_rewards))
        for score,length in zip(episodes_total_rewards, episodes_lengths):
            self.all_episodes_total_rewards += [score]
            self.all_episode_lengths += [length]
            self.all_avg_last_k += [self.get_last_k_episodes_mean()]
        self.total_steps += np.sum(episodes_lengths)
        self.done_episodes += len(episodes_total_rewards)
        if self.done_episodes - self.last_output_stats >= self.log_frequency:
            self.output_stats()

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

        plt.plot(xs, self.all_episodes_total_rewards, label='Episode-score')
        plt.plot(xs, self.all_avg_last_k, label='Score-last %d' % self.k)
        plt.ylabel('Score')
        plt.xlabel(x_label)
        plt.legend()
        plt.savefig(os.path.join(self.logdir, "Episode-scores.png"))
        plt.clf()

        for train_stats in self.agent_train_stats:
            self.agent_train_stats[train_stats].plot(os.path.join(self.logdir, train_stats+".png"))

        for hist_name in self.agent_histograms:
            plt.hist(self.agent_histograms[hist_name], bins='auto', label="Action histogram")
            plt.legend()
            plt.savefig(os.path.join(self.logdir, hist_name+".png"))
            plt.clf()

    def log_test(self, score):
        super(plt_logger, self).log_test(score)
        self.test_scores += [score]
        plt.plot( np.linspace(start=1, stop=max(2,len(self.all_avg_last_k)), num=len(self.test_scores)), self.test_scores)
        plt.ylabel('test_score-last %d'%self.k)
        plt.xlabel('Episode #')
        plt.savefig(os.path.join(self.logdir, "Test_scores.png"))
        plt.clf()

