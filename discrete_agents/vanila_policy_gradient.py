import torch
import numpy as np
import os
from dnn_models import MLP_softmax
from torch.distributions import Categorical

def get_action_vec(action, dim):
    res = np.zeros((dim, 1))
    res[action, 0] = 1
    return res

class vanila_policy_gradient_agent(object):
    def __init__(self, state_dim, action_dim, max_episodes, train=True):
        self.name = "Vanila-PG"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_episodes = max_episodes
        self.train=train
        self.batch_size = 1
        self.lr = 0.015
        self.discount = 0.99

        self.gs_num = 0
        self.completed_episodes = 0

        self.batch_episodes = []
        self.currently_building_episode = []
        self.device = torch.device("cpu")
        layers = [32,32]
        self.trainable_model = MLP_softmax(self.state_dim, self.action_dim, layers)

        self.optimizer = torch.optim.Adam(self.trainable_model.parameters(), lr=self.lr)
        self.optimizer.zero_grad()
        self.name += "_%s_lr[%.4f]_b[%d]"%(layers, self.lr, self.batch_size)

        self.last_action_log_prob = None

    def process_new_state(self, state):
        action_probs = self.trainable_model(torch.from_numpy(state).float())
        action_distribution = Categorical(action_probs)
        action_index = action_distribution.sample()
        self.last_action_log_prob = action_distribution.log_prob(action_index)
        action_index = action_index.item()

        return action_index

    def process_output(self, new_state, reward, is_finale_state):
        self.currently_building_episode += [(self.last_action_log_prob, reward)]
        if is_finale_state:
            self.batch_episodes += [self.currently_building_episode]
            self.currently_building_episode = []

        if len(self.batch_episodes) == self.batch_size:
            self._learn()
            self.gs_num += 1
            self.batch_episodes = []

    def _learn(self):
        loss = 0
        for rollout in self.batch_episodes:
            Rts = []
            Rt = 0
            for t in range(len(rollout) - 1, -1, -1):
                Rt = rollout[t][1] + self.discount * Rt
                Rts.insert(0, Rt)

            Rts = torch.tensor(Rts)
            Rts = (Rts - Rts.mean()) / (Rts.std())
            for t in range(len(rollout) - 1, -1, -1):
                log_prob, r = rollout[t]
                Rt = Rts[t]
                loss -= log_prob*Rt/len(self.batch_episodes)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


    def load_state(self, path):
        if os.path.exists(path):
            self.trainable_model.load_state_dict(torch.load(path))

    def save_state(self, path):
        torch.save(self.trainable_model.state_dict(), path)

    def get_stats(self):
        return "Gs: %d; LR: %.5f"%(self.gs_num, self.optimizer.param_groups[0]['lr'])


