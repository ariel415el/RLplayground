import torch
import random
from collections import deque
import numpy as np
import os
from dnn_models import MLP

def get_action_vec(action, dim):
    res = np.zeros((dim, 1))
    res[action, 0] = 1
    return res


class DQN_agent(object):
    def __init__(self, state_dim, action_dim, max_episodes, train = True):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_episodes = max_episodes
        self.train = train
        self.tau=1.
        self.lr = 0.001
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.discount = 0.99
        self.update_freq = 2
        self.batch_size = 32
        self.max_playback = 100000
        self.epsilon_decay = 0.996

        self.action_counter = 0
        self.completed_episodes = 0
        self.gs_num=0
        self.playback_deque = deque(maxlen=self.max_playback)

        self.device = torch.device("cpu")

        layers = [10,10]
        self.trainable_model = MLP(self.state_dim, self.action_dim, layers)
        with torch.no_grad():
            self.periodic_model = MLP(self.state_dim, self.action_dim, layers)
        self.update_net()

        self.optimizer = torch.optim.Adam(self.trainable_model.parameters(), lr=self.lr)
        self.optimizer.zero_grad()

        self.name = "DQN_%s_lr[%.4f]_b[%d]_tau[%.4f]_uf[%d]"%(str(layers), self.lr, self.batch_size, self.tau, self.update_freq)

    def update_net(self):
        for target_param, local_param in zip(self.periodic_model.parameters(), self.trainable_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def process_new_state(self, state):
        self.action_counter += 1
        if random.uniform(0,1) < self.epsilon:
            action_index =  random.randint(0, self.action_dim - 1)
        else:
            q_vals = self.trainable_model(torch.from_numpy(state).float())
            action_index = np.argmax(q_vals.detach().cpu().numpy())

        self.last_state = state
        self.last_action = action_index

        return action_index

    def process_output(self, new_state, reward, is_finale_state):
        self.playback_deque.append((self.last_state, self.last_action, new_state, reward, is_finale_state))
        if is_finale_state:
            self.epsilon = max(self.min_epsilon, self.epsilon*self.epsilon_decay)

        self._learn()
        if self.action_counter % self.update_freq == 0:
            self.update_net()

    def _learn(self):
        if len(self.playback_deque) >= self.batch_size:
            batch_arrays = np.array(random.sample(self.playback_deque, k=self.batch_size))
            prev_states = np.stack(batch_arrays[:, 0], axis=0)
            prev_actions = np.stack(batch_arrays[:, 1], axis=0)
            next_states = np.stack(batch_arrays[:, 2], axis=0)
            rewards = np.stack(batch_arrays[:, 3], axis=0)
            is_finale_states = np.stack(batch_arrays[:, 4], axis=0)

            with torch.no_grad():
                net_outs = self.periodic_model(torch.from_numpy(next_states).float())
                # net_outs = self.trainable_model(torch.tensor(next_states))

            target_values = torch.from_numpy(rewards)
            target_values[np.logical_not(is_finale_states)] += self.discount*net_outs.max(axis=1)[0][np.logical_not(is_finale_states)]
            # target_values = torch.tensor(rewards) + net_outs.max(axis=1)[0] * (torch.tensor(1 - is_finale_states))

            self.trainable_model.train()
            prev_net_outs = self.trainable_model(torch.from_numpy(prev_states).float())
            curr_q_vals = prev_net_outs[np.arange(prev_net_outs.shape[0]), prev_actions]
            # curr_q_vals = torch.matmul(prev_net_outs.view(-1,1, self.action_dim).float(), torch.tensor(prev_actions).float()).view(-1,1)
            # curr_q_vals = self.trainable_model.forward(torch.tensor(prev_states)).gather(1, torch.tensor(prev_actions))

            # loss = torch.nn.functional.mse_loss(curr_q_vals.double(), target_values.view(-1,1).double())
            loss = torch.nn.functional.mse_loss(curr_q_vals.double(), target_values.double())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.gs_num += 1

    def load_state(self, path):
        if os.path.exists(path):
            self.periodic_model.load_state_dict(torch.load(path))
            self.trainable_model.load_state_dict(torch.load(path))
        else:
            print("Couldn't find weights file")

    def save_state(self, path):
        torch.save(self.trainable_model.state_dict(), path)

    def get_stats(self):
        return "GS: %d, Epsilon: %.5f; LR: %.5f"%(self.gs_num, self.epsilon, self.optimizer.param_groups[0]['lr'])


