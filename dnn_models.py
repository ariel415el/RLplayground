import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class ATARIFeaturesExtraction(torch.nn.Module):
        def __init__(self, input_channels, conv_channels=[32, 64, 64]):
            super(ATARIFeaturesExtraction, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, conv_channels[0], kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, stride=1),
                nn.ReLU()
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return x

class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_sizes):
        super(MLP, self).__init__()
        layers = [torch.nn.Linear(input_dim, hidden_layer_sizes[0]), torch.nn.ReLU()]

        for i in range(1, len(hidden_layer_sizes)):
            layers += [torch.nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]), torch.nn.ReLU()]

        layers += [torch.nn.Linear(hidden_layer_sizes[-1], output_dim)]

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x

class MLP_softmax(MLP):
    def __init__(self, input_dim, output_dim, hidden_layer_sizes):
        MLP.__init__(self, input_dim, output_dim, hidden_layer_sizes)

    def forward(self, x):
        x = super().forward(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


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
