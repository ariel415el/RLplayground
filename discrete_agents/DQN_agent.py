import torch
import random
from collections import deque
import numpy as np
import os
from dnn_models import MLP
from utils import update_net, FastMemory
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device: ", device)


class conv_net(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(conv_net, self).__init__()
        self.state_dim = state_dim
        self.conv1 = torch.nn.Conv2d(self.state_dim[2], 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(self.state_dim[0]*self.state_dim[1] , 64)
        self.fc2 = torch.nn.Linear(64 , action_dim)

    def forward(self, x):
        x = x.float().view(-1, self.state_dim[2], self.state_dim[0], self.state_dim[1])
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = x.view(-1, self.state_dim[0]*self.state_dim[1])
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQN_agent(object):
    def __init__(self, state_dim, action_dim, train = True):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.train = train
        self.tau=0.5
        self.lr = 0.1
        self.epsilon = 0.10
        self.min_epsilon = 0.005
        self.discount = 0.99
        self.update_freq = 1
        self.batch_size = 32
        self.max_playback = 10000
        self.epsilon_decay = 0.996

        self.action_counter = 0
        self.completed_episodes = 0
        self.gs_num=0

        storage_sizes_and_types = [(self.state_dim, np.uint8), (1, np.uint8), (self.state_dim,np.float32), (1, np.float32), (1, bool)]
        self.playback_memory = FastMemory(self.max_playback, storage_sizes_and_types)

        layers = [64,64]
        self.trainable_model = conv_net(self.state_dim, self.action_dim).to(device)
        with torch.no_grad():
            self.target_model = copy.deepcopy(self.trainable_model)

        self.optimizer = torch.optim.Adam(self.trainable_model.parameters(), lr=self.lr)
        self.optimizer.zero_grad()

        self.name = "DQN_%s_lr[%.4f]_b[%d]_tau[%.4f]_uf[%d]"%(str(layers), self.lr, self.batch_size, self.tau, self.update_freq)

    def process_new_state(self, state):
        self.action_counter += 1
        if random.uniform(0,1) < self.epsilon:
            action_index =  random.randint(0, self.action_dim - 1)
        else:
            q_vals = self.trainable_model(torch.from_numpy(state).to(device))
            action_index = np.argmax(q_vals.detach().cpu().numpy())

        self.last_state = state
        self.last_action = action_index

        return action_index

    def process_output(self, new_state, reward, is_finale_state):
        self.playback_memory.add_sample((self.last_state, self.last_action, new_state, reward, is_finale_state))
        if is_finale_state:
            self.epsilon = max(self.min_epsilon, self.epsilon*self.epsilon_decay)

        self._learn()
        if self.action_counter % self.update_freq == 0:
            update_net(self.target_model, self.trainable_model, self.tau)

    def _learn(self):
        if len(self.playback_memory) >= self.batch_size:
            prev_states, prev_actions, next_states, rewards, is_finale_states = self.playback_memory.sample(self.batch_size ,device)

            with torch.no_grad():
                net_outs = self.target_model(next_states)

            target_values = rewards
            mask = (~is_finale_states).view(-1,1)
            target_values[mask] += self.discount*net_outs.max(axis=1)[0].reshape(-1,1)[mask]

            self.trainable_model.train()
            prev_net_outs = self.trainable_model(prev_states)
            curr_q_vals = torch.gather(prev_net_outs, dim=1, index = prev_actions.long())

            loss = torch.nn.functional.mse_loss(curr_q_vals, target_values)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.gs_num += 1

    def load_state(self, path):
        if os.path.exists(path):
            self.target_model.load_state_dict(torch.load(path))
            self.trainable_model.load_state_dict(torch.load(path))
        else:
            print("Couldn't find weights file")

    def save_state(self, path):
        torch.save(self.trainable_model.state_dict(), path)

    def get_stats(self):
        return "GS: %d, Epsilon: %.5f; LR: %.5f"%(self.gs_num, self.epsilon, self.optimizer.param_groups[0]['lr'])


