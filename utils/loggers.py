from _collections import deque
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
from time import time
import torch
import pickle


class logger(object):
    """Basic logger of training progress"""
    def __init__(self, log_frequency, logdir):
        self.logdir = logdir
        self.log_frequency = log_frequency
        self.episodes_scores = []
        self.score_scope_scores = []
        self.episodes_lengths = []
        self.train_start = time()
        self.total_steps = 0
        self.last_time = self.train_start
        self.agent_train_stats = {}
        self.agent_histograms = {}
        self.done_episodes = 0

    def add_costume_log(self, name, x, y):
        if name in self.agent_train_stats:
            self.agent_train_stats[name].add(x, y)
        else:
            self.agent_train_stats[name] = pyplot_scalar_writer(name, x, y)

    def add_histogram(self, name, values):
        if name not in self.agent_histograms:
            self.agent_histograms[name] = list(values)
        else:
            self.agent_histograms[name] += list(values)

    def log_episode(self, episode_score: int, score_scope_score: int,  episode_length: int):
        self.episodes_scores += [episode_score]
        self.score_scope_scores += [score_scope_score]
        self.episodes_lengths += [episode_length]
        self.total_steps += episode_length
        self.done_episodes += 1
        if self.done_episodes % self.log_frequency == 0:
            self.output_stats()

    def output_stats(self):
        time_passed = time() - self.train_start
        print('Episodes done: ', self.done_episodes)
        print("\t# Steps %d, time %d mins; score-scope %.2f:" % (self.total_steps, time_passed / 60, self.score_scope_scores[-1]))
        print("\t# steps/sec avg: %.3f " % (self.total_steps / time_passed))
        print("\t# Agent stats:", "; ".join([name+":%.5f"%self.agent_train_stats[name].ys[-1] for name in self.agent_train_stats]))
        self.last_time = time_passed

    def log_test(self, score):
        print("Test score: %.3f "%score)

    def pickle_episode_scores(self):
        f = open(os.path.join(self.logdir, "episode_scores.pkl"), 'wb')
        pickle.dump(self.episodes_scores, f)

class TB_logger(logger):
    """Outputs progress with tensorboard"""
    def __init__(self, log_frequency, logdir):
        super(TB_logger, self).__init__(log_frequency, logdir)
        from tensorboardX import SummaryWriter
        self.tb_writer = SummaryWriter(os.path.join(logdir,'tensorboard'))
        self.costume_log_global_steps = {} # TODO do this in another way

    def add_costume_log(self, name, x, y):
        if x is None:
            if name not in self.costume_log_global_steps:
                self.costume_log_global_steps[name] = 0
            x = self.costume_log_global_steps[name]
            self.costume_log_global_steps[name] += 1
        self.tb_writer.add_scalar(os.path.join("Agent",name), torch.tensor(y), global_step=x)

    def log_episode(self, episode_score, score_scope_score,  episode_length):
        super(TB_logger, self).log_episode(episode_score, score_scope_score,  episode_length)

        self.tb_writer.add_scalars('Env/Episode_score',{"Score": torch.tensor(episode_score),
                                                   'score-scope-avg': torch.tensor(score_scope_score)},
                                  global_step=self.done_episodes)

        self.tb_writer.add_scalar('Env/Episode-Length', torch.tensor(episode_length),
                                  global_step=self.done_episodes)

class plt_logger(logger):
    """Outputs progress with pyplot"""
    def __init__(self, log_frequency, logdir):
        super(plt_logger, self).__init__(log_frequency, logdir)
        os.makedirs(logdir, exist_ok=True)

    def output_stats(self, actor_stats=None, by_step=False):
        super(plt_logger, self).output_stats()
        xs = np.arange(1, len(self.episodes_scores) + 1)
        x_label = 'Episode #'
        if by_step:
            xs = np.cumsum(self.episodes_lengths)
            x_label = 'Step #'
        plt.plot(xs, self.episodes_lengths)
        plt.ylabel('Length')
        plt.xlabel(x_label)
        plt.savefig(os.path.join(self.logdir, "Episode-lengths.png"))
        plt.clf()

        plt.plot(xs, self.episodes_scores, label='Episode-score')
        plt.plot(xs, self.score_scope_scores, label='score-scope-avg')
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


class pyplot_scalar_writer(object):
    """This object allows a similar functionality as tensorboard's scalar writer only with pyplot
        it allows logging any custom scalr
    """
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
