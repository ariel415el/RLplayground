import torch
import random
from collections import deque
import numpy as np
import os
from torch import nn
from utils import update_net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device: ", device)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class D_Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, layer_dims=[400,200], init_w=3e-3, batch_norm=True):
        super(D_Actor, self).__init__()
        self.batch_norm = batch_norm
        self.fc1 = nn.Linear(nb_states, layer_dims[0])
        self.fc2 = nn.Linear(layer_dims[0], layer_dims[1])
        self.fc3 = nn.Linear(layer_dims[1], nb_actions)
        if batch_norm:
            self.bn1 = torch.nn.BatchNorm1d(layer_dims[0])
            self.bn2 = torch.nn.BatchNorm1d(layer_dims[1])

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = self.fc1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out


class D_Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, layer_dims=[400,200], init_w=3e-3, batch_norm=True):
        super(D_Critic, self).__init__()
        self.batch_norm = batch_norm
        self.fc1 = nn.Linear(nb_states, layer_dims[0])
        self.fc2 = nn.Linear(layer_dims[0] + nb_actions, layer_dims[1])
        self.fc3 = nn.Linear(layer_dims[1], 1)
        if batch_norm:
            self.bn1 = torch.nn.BatchNorm1d(layer_dims[0])
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, s, a):
        out = self.fc1(s)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out


class OUNoise:
    def __init__(self, size, mu=0, theta=.2, sigma=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu * np.ones(size)
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class DDPG(object):
    def __init__(self, state_dim, bounderies, max_episodes, train = True):
        self.state_dim = state_dim
        self.bounderies = bounderies
        self.action_dim = len(bounderies[0])
        self.max_episodes = max_episodes
        self.train = train
        self.tau=0.005
        self.actor_lr = 0.0005
        self.critic_lr = 0.005
        self.critic_weight_decay=0.001
        self.min_epsilon = 0.01
        self.discount = 0.99
        self.update_freq = 1
        self.batch_size = 100
        self.max_playback = 1000000
        self.random_process = OUNoise(self.action_dim)

        self.action_counter = 0
        self.completed_episodes = 0
        self.gs_num=0
        self.playback_deque = deque(maxlen=self.max_playback)

        layer_dims = [128,64]
        batch_norm=True
        self.trainable_actor = D_Actor(self.state_dim, self.action_dim, layer_dims, batch_norm).to(device)
        self.target_actor = D_Actor(self.state_dim, self.action_dim, layer_dims, batch_norm).to(device)

        self.trainable_critic = D_Critic(self.state_dim, self.action_dim, layer_dims, batch_norm).to(device)
        self.target_critic = D_Critic(self.state_dim, self.action_dim, layer_dims, batch_norm).to(device)

        update_net(self.target_actor, self.trainable_actor, 1)
        update_net(self.target_critic, self.trainable_critic, 1)

        self.actor_optimizer = torch.optim.Adam(self.trainable_actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.trainable_critic.parameters(), lr=self.critic_lr, weight_decay=self.critic_weight_decay)

        self.name = "DDPG_%s_"%str(layer_dims)
        if batch_norm:
            self.name += "BN_"
        self.name += "lr[%.4f]_b[%d]_tau[%.4f]_uf[%d]"%(self.actor_lr, self.batch_size, self.tau, self.update_freq)

    def process_new_state(self, state):
        self.action_counter += 1
        self.trainable_actor.eval()
        with torch.no_grad():
            state_torch = torch.from_numpy(state).to(device).float().view(1,-1)
            action = self.trainable_actor(state_torch).cpu().data.numpy()[0]
        self.trainable_actor.train()
        if self.train:
            # action += self.epsilon * self.random_process.sample()
            action += self.random_process.sample()

        self.last_state = state
        self.last_action = action

        # action = action.detach().cpu().numpy()
        action = np.clip(action, self.bounderies[0], self.bounderies[1])
        return action

    def process_output(self, new_state, reward, is_finale_state):
        if self.train:
            self.playback_deque.append((self.last_state, self.last_action, new_state, reward, is_finale_state))
            self._learn()
            if self.action_counter % self.update_freq == 0:
                update_net(self.target_actor, self.trainable_actor, self.tau)
                update_net(self.target_critic, self.trainable_critic, self.tau)
                # self.epsilon =max(self.min_epsilon, self.epsilon*self.epsilon_decay)
        # if is_finale_state:
        #     self.random_process.reset()

    def _learn(self):
        if len(self.playback_deque) > self.batch_size:
            batch_arrays = np.array(random.sample(self.playback_deque, k=self.batch_size))
            states = torch.from_numpy(np.stack(batch_arrays[:, 0], axis=0)).to(device).float()
            actions = torch.from_numpy(np.stack(batch_arrays[:, 1], axis=0)).to(device).float()
            next_states = torch.from_numpy(np.stack(batch_arrays[:, 2], axis=0)).to(device).float()
            rewards = torch.from_numpy(np.stack(batch_arrays[:, 3], axis=0)).to(device).float()
            is_finale_states = np.stack(batch_arrays[:, 4], axis=0)

            # update critic
            with torch.no_grad():
                next_target_action = self.target_actor(next_states)
                next_target_q_values = self.target_critic(next_states, next_target_action).view(-1)

                target_values = rewards
                mask = np.logical_not(is_finale_states)
                target_values[mask] += self.discount*next_target_q_values[mask]

            self.trainable_critic.train()
            self.critic_optimizer.zero_grad()
            q_values = self.trainable_critic(states, actions)
            loss = torch.nn.functional.mse_loss(q_values.view(-1), target_values)
            loss.backward()
            self.critic_optimizer.step()

            # update actor
            self.actor_optimizer.zero_grad()
            actions = self.trainable_actor(states)
            actor_obj = -self.trainable_critic(states, actions).mean()
            actor_obj.backward()
            self.actor_optimizer.step()

            self.gs_num += 1

    def load_state(self, path):
        if os.path.exists(path):
            # dict = torch.load(path)
            dict = torch.load(path, map_location=lambda storage, loc: storage)

            self.trainable_actor.load_state_dict(dict['actor'])
            self.trainable_critic.load_state_dict(dict['critic'])
        else:
            print("Couldn't find weights file")

    def save_state(self, path):
        dict = {'actor':self.trainable_actor.state_dict(), 'critic': self.trainable_critic.state_dict()}
        torch.save(dict, path)

    def get_stats(self):
        return "GS: %d; LR: a-%.5f\c-%.5f"%(self.gs_num, self.actor_optimizer.param_groups[0]['lr'],self.critic_optimizer.param_groups[0]['lr'])


