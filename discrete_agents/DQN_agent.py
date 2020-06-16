#####################################################################
# Code inspired from https://github.com/higgsfield/RL-Adventure.git #
#####################################################################

import torch
import random
from collections import deque
import numpy as np
import os
from dnn_models import *
from utils import update_net, FastMemory, PrioritizedMemory, ListMemory
import copy
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device: ", device)
from GenericAgent import GenericAgent

class new_DuelingDQN(nn.Module):
    def __init__(self, input_features):
        super(new_DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_features, 32, kernel_size=8, stride=4)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.constant_(self.conv2.bias, 0)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        nn.init.constant_(self.conv3.bias, 0)

        self.conv4 = nn.Conv2d(64, 1024, kernel_size=7, stride=1)
        nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')
        nn.init.constant_(self.conv4.bias, 0)
        # add comment
        self.fc_value = nn.Linear(512, 1)
        nn.init.kaiming_normal_(self.fc_value.weight, nonlinearity='relu')
        self.fc_advantage = nn.Linear(512, 4)
        nn.init.kaiming_normal_(self.fc_advantage.weight, nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # add comment
        x_value = x[:, :512, :, :].view(-1, 512)
        x_advantage = x[:, 512:, :, :].view(-1, 512)
        x_value = self.fc_value(x_value)
        x_advantage = self.fc_advantage(x_advantage)
        # add comment
        q_value = x_value + x_advantage.sub(torch.mean(x_advantage, 1)[:, None])
        return q_value


class LinearClassifier(nn.Module):
    def __init__(self, feature_extractor, num_outputs, hidden_layer_size):
        super(LinearClassifier, self).__init__()
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
        self.value_half = int(feature_extractor.features_space / 2)
        self.advantage_half = feature_extractor.features_space  - self.value_half

        self.value = nn.Sequential(
            nn.Linear(self.value_half, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(self.advantage_half, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, num_outputs)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        value_input, advantage_input = x[:,:self.value_half], x[:, self.value_half:]
        value = self.value(value_input)
        advantage = self.advantage(advantage_input)
        return value + advantage - advantage.mean()


def normalize_states(states):
    return (states - 127) / 255

class DQN_agent(GenericAgent):
    def __init__(self, state_dim, action_dim, hp=None, train=True, double_dqn=False, dueling_dqn=False, prioritized_memory=False, noisy_MLP=False):
        super(DQN_agent, self).__init__(train)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.train = train
        self.double_dqn=double_dqn
        self.dueling_dqn=dueling_dqn
        self.prioritized_memory=prioritized_memory
        self.noisy_MLP = noisy_MLP
        self.hp = {
            'tau': 1.0,
            'lr' : 0.00025,
            'min_epsilon' : 0.01,
            'discount' : 0.99,
            'update_freq' : 10000,
            'learn_freq': 1,
            'batch_size' : 32,
            'max_playback' : 1000000,
            'min_playback' : 50000,
            'epsilon': 1.0,
            'epsilon_decay' : 0.0001,
            'hiden_layer_size' : 512,
            'normalize_state':False
        }
        if hp is not None:
            self.hp.update(hp)

        self.action_counter = 0
        self.gs_num=0

        if type(self.state_dim) == tuple:
            feature_extractor = ConvNetFeatureExtracor(self.state_dim[0])
            state_dtype = np.uint8
        else:
            feature_extractor = LinearFeatureExtracor(self.state_dim, 64)
            state_dtype = np.float32
        if self.dueling_dqn:
            self.trainable_model = DuelingDQN(feature_extractor, self.action_dim, self.hp['hiden_layer_size']).to(device)
        elif self.noisy_MLP:
            self.trainable_model = NoisyMLP(feature_extractor, self.action_dim, self.hp['hiden_layer_size']).to(device)
        else:
            self.trainable_model = LinearClassifier(feature_extractor, self.action_dim, self.hp['hiden_layer_size']).to(device)
            # self.trainable_model = new_DuelingDQN(self.state_dim[0]).to(device)


        if self.prioritized_memory:
            storage_sizes_and_types = [(self.state_dim, state_dtype), (1, np.uint8), (self.state_dim, state_dtype), (1, np.float32), (1, bool)]
            self.playback_memory = PrioritizedMemory(self.hp['max_playback'], storage_sizes_and_types)
        else:
            self.playback_memory = ListMemory(self.hp['max_playback'])

        with torch.no_grad():
            self.target_model = copy.deepcopy(self.trainable_model)

        self.optimizer = torch.optim.Adam(self.trainable_model.parameters(), lr=self.hp['lr'])
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
        self.name += "lr[%.5f]_b[%d]_lf[%d]_uf[%d]"%(self.hp['lr'], self.hp['batch_size'], self.hp['learn_freq'], self.hp['update_freq'])

    def _get_cur_epsilon(self):
        return self.hp['min_epsilon'] + (self.hp['epsilon'] - self.hp['min_epsilon']) * np.exp(-1. * self.action_counter / self.hp['epsilon_decay'])

    def process_new_state(self, state):
        self.action_counter += 1
        if not self.noisy_MLP and self.train and random.uniform(0,1) < self._get_cur_epsilon():
            action_index = random.randint(0, self.action_dim - 1)
        else:
            torch_state = torch.from_numpy(np.array(state)).unsqueeze(0).to(device).float()
            if self.hp['normalize_state']:
                torch_state = normalize_states(torch_state)
            q_vals = self.trainable_model(torch_state)
            action_index = np.argmax(q_vals.detach().cpu().numpy())
        self.last_state = state
        self.last_action = action_index

        return action_index

    def process_output(self, new_state, reward, is_finale_state):
        self.playback_memory.add_sample((self.last_state, self.last_action, new_state, reward, is_finale_state))

        self._learn()
        if self.action_counter % self.hp['update_freq'] == 0:
            update_net(self.target_model, self.trainable_model, self.hp['tau'])

    def _learn(self):
        if len(self.playback_memory) >= max(self.hp['min_playback'], self.hp['batch_size']) and self.action_counter % self.hp['learn_freq'] == 0:
            if self.prioritized_memory:
                prev_states, prev_actions, next_states, rewards, is_finale_states, weights = self.playback_memory.sample(self.hp['batch_size'] ,device)
            else:
                prev_states, prev_actions, next_states, rewards, is_finale_states = self.playback_memory.sample(self.hp['batch_size'] ,device)

            prev_states = prev_states.float()
            next_states = next_states.float()
            if self.hp['normalize_state']:
                prev_states = normalize_states(prev_states)
                next_states = normalize_states(next_states)

            # Compute target
            target_net_outs = self.target_model(next_states)
            if self.double_dqn:
                trainable_net_outs = self.trainable_model(next_states)
                q_vals = target_net_outs.gather(1, trainable_net_outs.argmax(1).unsqueeze(1)) # uses trainalbe to choose actions and target to evaluate
                q_vals = q_vals.detach()
            else:
                q_vals = target_net_outs.max(axis=1)[0].reshape(-1,1)
            not_final = (1-is_finale_states.type(torch.float32).view(-1,1))
            target_values = rewards.view(-1,1) + self.hp['discount']*q_vals*not_final

            # Copute prediction
            self.trainable_model.train()
            prev_net_outs = self.trainable_model(prev_states)
            curr_q_vals = torch.gather(prev_net_outs, dim=1, index=prev_actions.long().unsqueeze(1))

            if self.prioritized_memory:
                weights = torch.from_numpy(weights).to(device)
                delta = (target_values - curr_q_vals)
                loss = ((delta**2)*weights).mean()
                delta = np.abs(delta.detach().cpu().numpy()).reshape(-1)
                self.playback_memory.update_priorities(delta)
            else:
                loss = (curr_q_vals - target_values).pow(2).mean()
                # loss = (target_values - curr_q_vals).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.gs_num += 1
            if self.noisy_MLP:
                self.trainable_model.reset_noise()
                self.target_model.reset_noise()

            self.reporter.update_agent_stats("Loss", self.action_counter, loss.item())

    def load_state(self, path):
        if os.path.exists(path):
            weights = torch.load(path, map_location=lambda storage, loc: storage)
            self.target_model.load_state_dict(weights)
            self.trainable_model.load_state_dict(weights)
        else:
            print("Couldn't find weights file")

    def save_state(self, path):
        torch.save(self.trainable_model.state_dict(), path)

    def get_stats(self):
        return "GS: %d, Epsilon: %.5f; LR: %.5f"%(self.gs_num, self._get_cur_epsilon(), self.optimizer.param_groups[0]['lr'])


