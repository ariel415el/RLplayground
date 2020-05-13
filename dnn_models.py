import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# DQN, vanila policy gradient and A2C
class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_sizes):
        super(MLP, self).__init__()
        layers = [torch.nn.Linear(input_dim, hidden_layer_sizes[0]), torch.nn.ReLU()]

        for i in range(1, len(hidden_layer_sizes)):
            layers += [torch.nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]), torch.nn.ReLU()]

        layers += [torch.nn.Linear(hidden_layer_sizes[-1], output_dim)]

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        # x = x.float()
        x = self.model(x)
        return x

class MLP_softmax(MLP):
    def __init__(self, input_dim, output_dim, hidden_layer_sizes):
        MLP.__init__(self, input_dim, output_dim, hidden_layer_sizes)

    def forward(self, x):
        x = super().forward(x)
        x = torch.nn.functional.softmax(x)
        return x


# ContinousActorCritic
class ContinousActorCritic(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(ContinousActorCritic, self).__init__()
        layers = [torch.nn.Linear(input_dim, hidden_dims[0]), torch.nn.ReLU()]
        for i in range(1, len(hidden_dims)):
            layers += [torch.nn.Linear(hidden_dims[i - 1], hidden_dims[i]), torch.nn.ReLU()]
        self.features = torch.nn.Sequential(*layers)

        self.mu_layer = torch.nn.Linear(hidden_dims[-1], output_dim)
        self.sigma_layer = torch.nn.Linear(hidden_dims[-1], output_dim)
        self.value_layer = torch.nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        features = self.features(x)

        mu = torch.nn.functional.tanh(self.mu_layer(features))
        sigma = torch.nn.functional.softplus(self.sigma_layer(features))
        value = self.value_layer(features)

        return mu, sigma, value

class ContinousActorCritic_2(nn.Module):
    def __init__(self, state_dim, action_dim, layers_dims):
        super(ContinousActorCritic_2, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, layers_dims[0]),
                nn.Tanh(),
                nn.Linear(layers_dims[0], layers_dims[1]),
                nn.Tanh(),
                nn.Linear(layers_dims[1], action_dim),
                nn.Tanh()
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, layers_dims[0]),
                nn.Tanh(),
                nn.Linear(layers_dims[0], layers_dims[1]),
                nn.Tanh(),
                nn.Linear(layers_dims[1], 1)
                )

    def get_value(self, state):
        return self.critic(state)

    def get_mu(self, state):
        return self.actor(state)

    def forward(self, x):
        raise NotImplementedError


# DPG
class DeterministicActorCritic(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(DeterministicActorCritic, self).__init__()
        layers = [torch.nn.Linear(input_dim, hidden_dims[0]), torch.nn.ReLU()]
        for i in range(1, len(hidden_dims)):
            layers += [torch.nn.Linear(hidden_dims[i - 1], hidden_dims[i]), torch.nn.ReLU()]
        self.features = torch.nn.Sequential(*layers)

        self.mu_layer = torch.nn.Linear(hidden_dims[-1], output_dim)
        self.value_layer = torch.nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        features = self.features(x)

        mu = torch.nn.functional.tanh(self.mu_layer(features))
        sigma = torch.nn.functional.softplus(self.sigma_layer(features))
        value = self.value_layer(features)

        return mu, sigma, value

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

class TD3_paper_actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1):
        super(TD3_paper_actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class TD3_paper_critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TD3_paper_critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1