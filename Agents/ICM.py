import torch
from torch import nn
from Agents.dnn_models import Swish


class Forward_module(nn.Module):
    def __init__(self, action_dim, hidden_dim, activation=nn.ReLU(inplace=True)):
        super(Forward_module, self).__init__()
        action_embedd_dim=128
        self.action_encoder = nn.Embedding(action_dim, action_embedd_dim)
        self.layers = nn.Sequential(nn.Linear(hidden_dim + action_embedd_dim, hidden_dim),
                                            activation,
                                            nn.Linear(hidden_dim, hidden_dim),
                                            activation,
                                            nn.Linear(hidden_dim, hidden_dim),
                                            )

    def forward(self, feature_maps, actions):
        actions = self.action_encoder(actions)
        ns_latent = torch.cat((feature_maps, actions), dim=-1)
        ns_latent = self.layers(ns_latent)
        return ns_latent

class Inverse_module(nn.Module):
    def __init__(self, action_dim, hidden_dim, activation=nn.ReLU(inplace=True)):
        super(Inverse_module, self).__init__()
        self.layers = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim),
                                            activation,
                                            nn.Linear(hidden_dim, hidden_dim),
                                            activation,
                                            nn.Linear(hidden_dim, action_dim)
                                            )  # softmax in cross entropy loss

    def forward(self, s_featues, ns_featues):
        input = torch.cat((s_featues, ns_featues), dim=-1)
        return self.layers(input)

class ICM_module(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, activation=nn.ReLU(inplace=True)):
        super(ICM_module, self).__init__()
        self.state_feature_extractor = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                                     activation,
                                                     nn.Linear(hidden_dim, hidden_dim),
                                                     activation,
                                                     nn.Linear(hidden_dim, hidden_dim)
                                                     )
        self.inverse_module = Inverse_module(action_dim, hidden_dim, activation)
        self.forward_module = Forward_module(action_dim, hidden_dim, activation)


class ICM(object):
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.001, intrinsic_reward_scale=1.0, beta=0.2):
        super(ICM, self).__init__()
        self.action_dim = action_dim
        self._beta = beta
        self._intrinsic_reward_scale = intrinsic_reward_scale
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.module = ICM_module(state_dim, action_dim, hidden_dim, activation=nn.ReLU())
        self.curiosity_optimizer = torch.optim.Adam(self.module.parameters(), lr=lr)

    def get_intrinsic_reward(self, states, next_states, actions):
        s_featues = self.module.state_feature_extractor(states)
        ns_featues = self.module.state_feature_extractor(next_states)

        # inverse loss
        estimated_actions = self.module.inverse_module(s_featues, ns_featues)
        loss_i = self.cross_entropy_loss(estimated_actions, actions)

        # Forward loss
        estimated_ns_features = self.module.forward_module(s_featues, actions)
        # features_dists = (0.5*(ns_featues - estimated_ns_features).pow(2)).sum(1)
        features_dists = 0.5*(ns_featues - estimated_ns_features).norm(2, dim=-1).pow(2)
        loss_f = features_dists.mean()

        # Intrinsic reward
        intrisic_rewards = self._intrinsic_reward_scale*features_dists

        # Optimize
        curiosity_loss = (1-self._beta)*loss_i + self._beta*loss_f
        self.curiosity_optimizer.zero_grad()
        curiosity_loss.backward()
        self.curiosity_optimizer.step()

        self.debug_loss = curiosity_loss.item()

        return intrisic_rewards.detach().cpu().numpy()

    def get_last_debug_loss(self):
        return self.debug_loss