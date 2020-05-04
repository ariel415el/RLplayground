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
    def __init__(self, state_dim, action_dim):
        super(ContinousActorCritic_2, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Tanh()
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
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
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=200, init_w=3e-3):
        super(D_Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.bn1 = torch.nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = torch.nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out


class D_Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=200, init_w=3e-3):
        super(D_Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.bn1 = torch.nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, s, a):
        out = self.fc1(s)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out