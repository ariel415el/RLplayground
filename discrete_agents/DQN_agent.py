#####################################################################
# Code inspired from https://github.com/higgsfield/RL-Adventure.git #
#####################################################################

import torch
import random
from collections import deque
import numpy as np
import os
from dnn_models import *
from utils import update_net, FastMemory, PrioritizedMemory
import copy
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device: ", device)


class MLP(nn.Module):
    def __init__(self, feature_extractor, num_outputs, hidden_layer_size):
        super(MLP, self).__init__()
        self.feature_extractor = feature_extractor
        self.linear1 = nn.Linear(self.feature_extractor.features_space, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, num_outputs)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class NoisyMLP(nn.Module):
    def __init__(self, feature_extractor, num_outputs, hidden_layer_size):
        super(NoisyMLP, self).__init__()
        self.feature_extractor = feature_extractor
        self.noisy1 = NoisyLinear(self.feature_extractor.features_space, hidden_layer_size)
        self.noisy2 = NoisyLinear(hidden_layer_size, num_outputs)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.nn.functional.relu(self.noisy1(x))
        x = self.noisy2(x)
        return x

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

class DuelingDQN(nn.Module):
    def __init__(self, feature_extractor, num_outputs, hidden_layer_size):
        super(DuelingDQN, self).__init__()

        self.feature_extractor = feature_extractor

        self.advantage = nn.Sequential(
            nn.Linear(feature_extractor.features_space, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, num_outputs)
        )

        self.value = nn.Sequential(
            nn.Linear(feature_extractor.features_space, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()


class DQN_agent(object):
    def __init__(self, state_dim, action_dim, train = True, double_dqn=False, dueling_dqn=False, prioritized_memory=False, noisy_MLP=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.train = train
        self.double_dqn=double_dqn
        self.dueling_dqn=dueling_dqn
        self.prioritized_memory=prioritized_memory
        self.noisy_MLP = noisy_MLP
        self.tau=1.0
        self.lr = 0.0001
        self.epsilon = 1.0
        self.min_epsilon = 0.005
        self.discount = 0.99
        self.update_freq = 1000
        self.batch_size = 32
        self.max_playback = 100000
        self.min_playback = 10000
        self.epsilon_decay = 0.996

        self.action_counter = 0
        self.completed_episodes = 0
        self.gs_num=0

        if type(self.state_dim) == tuple:
            feature_extractor = ConvNetFeatureExtracor(self.state_dim[0])
            state_dtype = np.uint8
            state_dtype = np.float32
        else:
            feature_extractor = LinearFeatureExtracor(self.state_dim, 64)
            state_dtype = np.float32
        hiden_layer_size = 512
        if self.dueling_dqn:
            self.trainable_model = DuelingDQN(feature_extractor, self.action_dim, hiden_layer_size).to(device)
        elif self.noisy_MLP:
            self.trainable_model = NoisyMLP(feature_extractor, self.action_dim, hiden_layer_size).to(device)
        else:
            self.trainable_model = MLP(feature_extractor, self.action_dim, hiden_layer_size).to(device)

        storage_sizes_and_types = [(self.state_dim, state_dtype), (1, np.uint8), (self.state_dim, state_dtype), (1, np.float32), (1, bool)]
        if self.prioritized_memory:
            self.playback_memory = PrioritizedMemory(self.max_playback, storage_sizes_and_types)
        else:
            self.playback_memory = FastMemory(self.max_playback, storage_sizes_and_types)

        with torch.no_grad():
            self.target_model = copy.deepcopy(self.trainable_model)

        self.optimizer = torch.optim.Adam(self.trainable_model.parameters(), lr=self.lr)
        self.optimizer.zero_grad()

        self.name = ""
        if self.double_dqn:
            self.name += "DobuleDQN-"
        if self.dueling_dqn:
            self.name += "DuelingDqn-"
        if self.prioritized_memory:
            self.name += "PriorityMemory-"
        if self.noisy_MLP:
            self.name += "NoisyNetwork-"
        else:
            self.name += "Dqn-"
        self.name += "lr[%.5f]_b[%d]_tau[%.4f]_uf[%d]"%(self.lr, self.batch_size, self.tau, self.update_freq)


    def process_new_state(self, state):
        self.action_counter += 1
        if not self.noisy_MLP and self.train and random.uniform(0,1) < self.epsilon:
            action_index =  random.randint(0, self.action_dim - 1)
        else:
            q_vals = self.trainable_model(torch.from_numpy(state).unsqueeze(0).to(device).float())
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
        if len(self.playback_memory) >= max(self.min_playback, self.batch_size):
            if self.prioritized_memory:
                prev_states, prev_actions, next_states, rewards, is_finale_states, weights = self.playback_memory.sample(self.batch_size ,device)
            else:
                prev_states, prev_actions, next_states, rewards, is_finale_states = self.playback_memory.sample(self.batch_size ,device)

            # Compute target
            target_net_outs = self.target_model(next_states)
            if self.double_dqn:
                trainable_net_outs = self.trainable_model(next_states)
                q_vals = target_net_outs.gather(1, trainable_net_outs.argmax(1).unsqueeze(1)) # uses trainalbe to choose actions and target to evaluate
                q_vals = q_vals.detach()
            else:
                q_vals = target_net_outs.max(axis=1)[0].reshape(-1,1)

            target_values = self.discount*q_vals*(1-is_finale_states.type(torch.float32))

            # Copute prediction
            self.trainable_model.train()
            prev_net_outs = self.trainable_model(prev_states)
            curr_q_vals = torch.gather(prev_net_outs, dim=1, index=prev_actions.long())

            if self.prioritized_memory:
                weights = torch.from_numpy(weights).to(device)
                delta = (target_values - curr_q_vals)
                loss = ((delta**2)*weights).mean()
                delta = np.abs(delta.detach().cpu().numpy()).reshape(-1)
                self.playback_memory.update_priorities(delta)
            else:
                loss = (target_values - curr_q_vals).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.gs_num += 1
            if self.noisy_MLP:
                self.trainable_model.reset_noise()
                self.target_model.reset_noise()


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


