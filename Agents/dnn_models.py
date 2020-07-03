import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributions as D


class NoisyLinear(nn.Module):
    # implements https://arxiv.org/abs/1706.10295
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(torch.autograd.Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(torch.autograd.Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class LinearFeatureExtracor(nn.Module):
    def __init__(self, num_inputs, num_outputs, activation=torch.relu):
        super(LinearFeatureExtracor, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        self.features_space = num_outputs
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.linear(x))
        return x


class ConvNetFeatureExtracor(nn.Module):
    ## Assumes input is input_channelsx84x84
    def __init__(self, input_channels):
        super(ConvNetFeatureExtracor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.features_space = 64*7*7

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.view(-1, self.features_space)
        return x


class DiscreteActor(nn.Module):
    def __init__(self, feature_extractor, action_dim, hidden_layers):
        super(DiscreteActor, self).__init__()
        self.features = feature_extractor

        layers = []
        last_features_space = self.features.features_space
        for layer_size in hidden_layers:
            layers += [nn.Linear(last_features_space, layer_size), nn.Tanh()]
            last_features_space = layer_size
        layers += [nn.Linear(last_features_space, action_dim), nn.Softmax(dim=1)]
        self.head = nn.Sequential(*layers)

    def get_dist(self, features):
        probs = self.head(features)
        dist = D.Categorical(probs)
        return dist

    def forward(self, x):
        features = self.features(x)
        dist = self.get_dist(features)
        return dist


class CountinousActor(nn.Module):
    def __init__(self, feature_extractor, action_dim, hidden_layers):
        super(CountinousActor, self).__init__()
        self.action_dim = action_dim
        self.features = feature_extractor
        layers = []
        last_features_space = self.features.features_space
        for layer_size in hidden_layers:
            layers += [nn.Linear(last_features_space, layer_size), nn.ReLU()]
            last_features_space = layer_size
        layers += [nn.Linear(last_features_space, action_dim), nn.Tanh()]
        self.log_sigma = nn.Parameter(torch.zeros(1, action_dim), requires_grad=True)
        self.head = nn.Sequential(*layers)

    def get_dist(self, features):
        mu = self.head(features)
        # dist_old = D.Normal(mu, self.log_sigma.exp())
        dist = D.multivariate_normal.MultivariateNormal(mu, torch.diag_embed(self.log_sigma.exp()))

        return dist

    def forward(self, x):
        features = self.features(x)
        dist = self.get_dist(features)
        return dist


class Critic(nn.Module):
    def __init__(self, feature_extractor, hidden_layers):
        super(Critic, self).__init__()
        self.features = feature_extractor
        layers = []
        last_features_space = self.features.features_space
        for layer_size in hidden_layers:
            layers += [nn.Linear(last_features_space, layer_size), nn.ReLU()]
            last_features_space = layer_size
        layers += [nn.Linear(last_features_space, 1)]
        self.head = nn.Sequential(*layers)

    def get_value(self, features):
        return self.head(features)

    def forward(self, x):
        features = self.features(x)
        value = self.get_value(features)
        return value

class ActorCriticModel(nn.Module):
    def __init__(self, feature_extractor, action_dim, hidden_layers, discrete=True):
        super(ActorCriticModel, self).__init__()
        # action mean range -1 to 1
        self.features = feature_extractor
        if discrete:
            self.actor = DiscreteActor(self.features, action_dim, hidden_layers)
        else:
            self.actor = CountinousActor(self.features, action_dim, hidden_layers)
        self.critic = Critic(self.features, hidden_layers)

    def get_action_dist(self, x):
        return self.actor(x)

    def forward(self, x):
        features = self.features(x)
        dist = self.actor.get_dist(features)
        value = self.critic.get_value(features)
        return dist, value