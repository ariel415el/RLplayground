import torch
import random
from collections import deque
import numpy as np
import os
from dnn_models import D_Actor, D_Critic
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device: ", device)

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

def get_action_vec(action, dim):
    res = np.zeros((dim, 1))
    res[action, 0] = 1
    return res

def update_net(model_to_change, reference_model, tau):
    for target_param, local_param in zip(model_to_change.parameters(), reference_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class DDPG(object):
    def __init__(self, state_dim, bounderies, max_episodes, train = True):
        self.state_dim = state_dim
        self.bounderies = bounderies
        self.action_dim = len(bounderies[0])
        self.max_episodes = max_episodes
        self.train = train
        self.tau=0.001
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.critic_weight_decay=0.001
        self.min_epsilon = 0.01
        self.discount = 0.99
        self.update_freq = 1
        self.batch_size = 1096
        self.max_playback = 1000000
        self.random_process = OUNoise(self.action_dim)

        self.action_counter = 0
        self.completed_episodes = 0
        self.gs_num=0
        self.playback_deque = deque(maxlen=self.max_playback)

        layer_dims = [400,200]
        batch_norm=False
        self.trainable_actor = D_Actor(self.state_dim, self.action_dim, layer_dims, batch_norm).to(device)
        self.target_actor = D_Actor(self.state_dim, self.action_dim, layer_dims, batch_norm).to(device)

        self.trainable_critic = D_Critic(self.state_dim, self.action_dim, layer_dims, batch_norm).to(device)
        self.target_critic = D_Critic(self.state_dim, self.action_dim, layer_dims, batch_norm).to(device)

        update_net(self.target_actor, self.trainable_actor, 1)
        update_net(self.target_critic, self.trainable_critic, 1)

        self.actor_optimizer = torch.optim.Adam(self.trainable_actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.trainable_critic.parameters(), lr=self.critic_lr, weight_decay=self.critic_weight_decay)

        # self.name = "DDPG_%s_"%str(layer_dims)
        # if batch_norm:
        #     self.name += "_BN_"
        self.name = "DDPG_lr[%.4f]_b[%d]_tau[%.4f]_uf[%d]"%(self.actor_lr, self.batch_size, self.tau, self.update_freq)

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
        if is_finale_state:
            self.random_process.reset()

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


