import torch
from utils import ListMemory
import numpy as np
import os
from torch import nn
from utils import update_net
from dnn_models import ConvNetFeatureExtracor, LinearFeatureExtracor
import copy
from GenericAgent import GenericAgent

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
        # self.fc2 = nn.Linear(layer_dims[0] + nb_actions, layer_dims[1])
        self.fc2_state = nn.Linear(layer_dims[0], layer_dims[1])
        self.fc2_action = nn.Linear(nb_actions, layer_dims[1])
        self.fc3 = nn.Linear(layer_dims[1], 1)
        if batch_norm:
            self.bn1 = torch.nn.BatchNorm1d(layer_dims[0])
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        # self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2_action.weight.data.uniform_(*hidden_init(self.fc2_action))
        self.fc2_state.weight.data.uniform_(*hidden_init(self.fc2_state))
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, s, a):
        out = self.fc1(s)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.relu(out)
        # out = self.fc2(torch.cat([out, a], 1))
        out = self.fc2_action(a) + self.fc2_state(out)
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


class DDPG(GenericAgent):
    def __init__(self, state_dim, bounderies, hp, train = True):
        super(DDPG, self).__init__(train)
        self.state_dim = state_dim
        self.bounderies = bounderies
        self.action_dim = len(bounderies[0])
        self.train = train
        self.hp = {
            'tau':0.05,
            'actor_lr':0.0005,
            'critic_lr':0.005,
            'critic_weight_decay':0.001,
            'min_epsilon':0.01,
            'discount':0.99,
            'update_freq':1,
            'batch_size':100,
            'max_playback':1000000,
            'min_playback':0,
            'learn_freq':1,
            'layer_dims':[128,64],
            'batch_norm':True
        }
        self.hp.update(hp)
        self.playback_memory = ListMemory(self.hp['max_playback'])

        self.random_process = OUNoise(self.action_dim)

        # if type(self.state_dim) == tuple:
        #     feature_extractor = ConvNetFeatureExtracor(self.state_dim[0])
        #     state_dtype = np.uint8
        # else:
        #     feature_extractor = LinearFeatureExtracor(self.state_dim, 64)
        #     state_dtype = np.float32

        self.trainable_actor = D_Actor(self.state_dim, self.action_dim, self.hp['layer_dims'], self.hp['batch_norm']).to(device)
        self.trainable_critic = D_Critic(self.state_dim, self.action_dim, self.hp['layer_dims'], self.hp['batch_norm']).to(device)

        with torch.no_grad():
            self.target_actor = copy.deepcopy(self.trainable_actor)
            self.target_critic = copy.deepcopy(self.trainable_critic)

        self.actor_optimizer = torch.optim.Adam(self.trainable_actor.parameters(), lr=self.hp['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.trainable_critic.parameters(), lr=self.hp['critic_lr'], weight_decay=self.hp['critic_weight_decay'])

        self.action_counter = 0
        self.completed_episodes = 0
        self.gs_num=0

        self.name = "DDPG_%s_"%str(self.hp['layer_dims'])
        if self.hp['batch_norm']:
            self.name += "BN_"
        self.name += "lr[%.4f]_b[%d]_tau[%.4f]_uf[%d]_lf[%d]"%(self.hp['actor_lr'], self.hp['batch_size'], self.hp['tau'], self.hp['update_freq'], self.hp['learn_freq'])

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

        action = np.clip(action, self.bounderies[0], self.bounderies[1])
        return action

    def process_output(self, new_state, reward, is_finale_state):
        if self.train:
            self.playback_memory.add_sample((self.last_state.astype(np.float32), self.last_action, new_state.astype(np.float32), reward, is_finale_state))
            self._learn()
            if self.action_counter % self.hp['update_freq'] == 0:
                update_net(self.target_actor, self.trainable_actor, self.hp['tau'])
                update_net(self.target_critic, self.trainable_critic, self.hp['tau'])
                # self.epsilon =max(self.min_epsilon, self.epsilon*self.epsilon_decay)
        # if is_finale_state:
        #     self.random_process.reset()

    def _learn(self):
        if len(self.playback_memory) >= max(self.hp['min_playback'], self.hp['batch_size']) and self.action_counter % self.hp['learn_freq'] == 0:
            states, actions, next_states, rewards, is_finale_states = self.playback_memory.sample(self.hp['batch_size'], device)

            # update critic
            with torch.no_grad():
                next_target_action = self.target_actor(next_states)
                next_target_q_values = self.target_critic(next_states, next_target_action)

                not_final = (1 - is_finale_states.float().view(-1, 1))
                target_values = rewards.view(-1, 1) + self.hp['discount'] * next_target_q_values * not_final

            self.critic_optimizer.zero_grad()
            q_values = self.trainable_critic(states, actions)
            loss = (0.5*(q_values - target_values).pow(2)).mean()
            loss.backward()
            self.critic_optimizer.step()
            self.reporter.update_agent_stats("Critic-Loss", self.action_counter, loss.item())

            # update actor
            self.actor_optimizer.zero_grad()
            actions = self.trainable_actor(states)
            actor_obj = -self.trainable_critic(states, actions).mean()
            actor_obj.backward()
            self.actor_optimizer.step()
            self.reporter.update_agent_stats("Actor-Loss", self.action_counter, actor_obj.item())

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


