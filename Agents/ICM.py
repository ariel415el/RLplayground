################################################################
### Inspires from https://github.com/adik993/ppo-pytorch.git ###
################################################################
import torch
from torch import nn

class ICM_module(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ICM_module, self).__init__()
        self.state_feature_extractor = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                                     nn.ReLU(inplace=True),
                                                     nn.Linear(hidden_dim, hidden_dim),
                                                     nn.ReLU(inplace=True),
                                                     nn.Linear(hidden_dim, hidden_dim)
                                                     )
        self.inverted_model = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, action_dim)
                                            )  # softmax in cross entropy loss
        self.forward_module = nn.Sequential(nn.Linear(hidden_dim + action_dim, hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, hidden_dim)
                                            )

class ICM(object):
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.001, intrinsic_reward_scale=1.0, beta=0.2):
        super(ICM, self).__init__()
        self.action_dim = action_dim
        self._beta = beta
        self._intrinsic_reward_scale = intrinsic_reward_scale
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.module = ICM_module(state_dim, action_dim, hidden_dim)
        self.curiosity_optimizer = torch.optim.Adam(self.module.parameters(), lr=lr)

    def get_intrinsic_loss(self, states, next_states, actions):
        s_featues = self.module.state_feature_extractor(states)
        ns_featues = self.module.state_feature_extractor(next_states)
        action_1hot_vecs = torch.zeros((actions.shape[0], self.action_dim)).to(actions.device)
        action_1hot_vecs[torch.arange(actions.shape[0]) , actions] = 1

        estimated_actions = self.module.inverted_model(torch.cat((s_featues, ns_featues),dim=-1))
        estimated_ns_features = self.module.forward_module(torch.cat((s_featues, action_1hot_vecs), dim=-1))

        loss_i = self.cross_entropy_loss(estimated_actions, actions)
        features_dists = (0.5*(ns_featues - estimated_ns_features).pow(2)).mean(1)
        loss_f = features_dists.mean()

        intrisic_rewards = self._intrinsic_reward_scale*features_dists
        curiosity_loss = (1-self._beta)*loss_i + self._beta*loss_f

        self.curiosity_optimizer.zero_grad()
        curiosity_loss.backward()
        self.curiosity_optimizer.step()

        self.debug_loss = curiosity_loss.item()

        return intrisic_rewards.detach().cpu().numpy()

    def get_last_debug_loss(self):
        return self.debug_loss