import numpy as np
import os
from dnn_models import *
torch.manual_seed(0)
import torch.distributions as D


class A2C(object):
    def __init__(self, state_dim, action_bounderies, max_episodes, train=True, critic_objective="Monte-Carlo"):
        self.name = 'c-actor-critic'
        self.state_dim = state_dim
        self.action_bounderies = action_bounderies
        self.action_dim = len(action_bounderies)
        self.max_episodes = max_episodes
        self.critic_objective = critic_objective
        self.train= train
        self.batch_size = 1
        self.discount = 0.99
        self.lr = 0.02
        self.epsilon_decay = 0.995

        self.device = torch.device("cpu")
        hidden_dims=[32]
        self.actor_critic = ContinousActorCritic(state_dim, self.action_dim, hidden_dims)

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.lr, betas = (0.9, 0.999))
        self.optimizer.zero_grad()
        self.gs_num = 0
        self.completed_episodes = 0

        self.batch_episodes = []
        self.currently_building_episode = []
        self.name += "_%s_lr[%.4f]_b[%d]_CO-%s"%(str(hidden_dims), self.lr, self.batch_size, self.critic_objective)


    def clip_action(self, action):
        return  np.clip(action, self.action_bounderies[0], self.action_bounderies[0])

    def get_action(self, state):
        mu, sigma, state_value = self.actor_critic(torch.from_numpy(state).float())
        action_distribution = D.multivariate_normal.MultivariateNormal(mu, torch.diag(sigma))
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        action = self.clip_action(action.numpy())
        return action, log_prob, state_value

    def process_new_state(self, state):
        action_index, self.last_action_log_prob, self.last_state_value = self.get_action(state)
        return action_index

    def process_output(self, new_state, reward, is_finale_state):
        self.currently_building_episode += [(self.last_action_log_prob, self.last_state_value, reward)]
        if is_finale_state:
            self.batch_episodes += [self.currently_building_episode]
            self.currently_building_episode = []

        if len(self.batch_episodes) == self.batch_size:
            self._learn()
            self.gs_num += 1
            self.batch_episodes = []

    def _learn(self):
        loss = 0
        for rollout in self.batch_episodes:
            qsa_estimates = []
            if self.critic_objective=="Monte-Carlo":
                qsa = 0
                for t in range(len(rollout) - 1, -1, -1):
                    qsa = rollout[t][2] + self.discount * qsa
                    qsa_estimates.insert(0, qsa)
                qsa_estimates = torch.tensor(qsa_estimates)
                qsa_estimates = (qsa_estimates - qsa_estimates.mean()) / (qsa_estimates.std())
            elif self.critic_objective=="TD":
                with torch.no_grad():
                    torch_rollout = torch.tensor(rollout)
                    qsa_estimates = torch_rollout[:, 2]
                    qsa_estimates[:-1] += torch_rollout[1:, 1]
            for t in range(len(rollout)):
                log_prob, s_value, r = rollout[t]
                with torch.no_grad():
                    At = qsa_estimates[t] - s_value.item()
                actor_obj = -log_prob*At / len(self.batch_episodes)
                critic_obj = torch.nn.functional.smooth_l1_loss(s_value, qsa_estimates[t])

                loss += actor_obj + critic_obj
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def load_state(self, path):
        if os.path.exists(path):
            self.actor_critic.load_state_dict(torch.load(path))

    def save_state(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def get_stats(self):
        return "Gs: %d; LR: %.5f"%(self.gs_num, self.optimizer.param_groups[0]['lr'])



