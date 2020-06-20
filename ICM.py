################################################################
### Inspires from https://github.com/adik993/ppo-pytorch.git ###
################################################################
import torch
from torch import nn

class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, _intrinsic_reward_scale=1.0, _beta=0.2):
        super(ICM, self).__init__()
        self.action_dim = action_dim
        self._beta = _beta
        self._intrinsic_reward_scale = _intrinsic_reward_scale
        self.state_features = hidden_dim
        self.state_feature_extractor = nn.Sequential(nn.Linear(state_dim, self.state_features),
                                                     nn.ReLU(inplace=True),
                                                     nn.Linear(self.state_features, self.state_features)
                                                    )
        self.inverted_model = nn.Sequential(nn.Linear(2*self.state_features, self.state_features),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.state_features, self.state_features),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.state_features, action_dim)
                                            ) # softmax in cross entropy loss
        self.forward_module = nn.Sequential(nn.Linear(self.state_features + action_dim, self.state_features),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.state_features, self.state_features),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.state_features, self.state_features)
                                            )
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def compute_reward_and_loss(self, states, next_states, actions):
        s_featues = self.state_feature_extractor(states)
        ns_featues = self.state_feature_extractor(next_states)
        action_vecs = torch.zeros((actions.shape[0], self.action_dim)).to(actions.device)
        action_vecs[torch.arange(actions.shape[0]) , actions] = 1

        estimated_actions = self.inverted_model(torch.cat((s_featues, ns_featues),dim=-1))
        estimated_ns_features = self.forward_module(torch.cat((s_featues, action_vecs), dim=-1))

        loss_i = self.cross_entropy_loss(estimated_actions, actions)
        features_dists = (0.5*(ns_featues - estimated_ns_features).pow(2)).mean(1)
        loss_f = features_dists.mean()

        intrisic_rewards = self._intrinsic_reward_scale*features_dists.detach()
        curiosity_loss = (1-self._beta)*loss_i + self._beta*loss_f
        return intrisic_rewards, curiosity_loss