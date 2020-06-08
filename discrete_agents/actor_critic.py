import numpy as np
import os
from dnn_models import *
torch.manual_seed(0)
from torch.distributions import Categorical

def get_action_vec(action, dim):
    res = np.zeros((dim, 1))
    res[action, 0] = 1
    return res

class Descrete_PPO_actor_critic(nn.Module):
    def __init__(self, state_dim, action_dim, layers_dims):
        super(Descrete_PPO_actor_critic, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, layers_dims[0]),
                nn.Tanh(),
                nn.Linear(layers_dims[0], layers_dims[1]),
                nn.Tanh(),
                nn.Linear(layers_dims[1], action_dim),
                nn.Softmax(dim=1)
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, layers_dims[0]),
                nn.Tanh(),
                nn.Linear(layers_dims[0], layers_dims[1]),
                nn.Tanh(),
                nn.Linear(layers_dims[1], 1)
                )

    def get_probs(self, x):
        probs = self.actor(x)
        return probs

    def forward(self, x):
        probs = self.actor(x)
        value = self.critic(x)
        return probs, value

class actor_critic_agent(object):
    def __init__(self, state_dim, action_dim, hp, train=True, critic_objective="Monte-Carlo"):
        self.name = 'actor-critic'
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.critic_objective = critic_objective
        self.train= train
        self.hp = {
            'batch_size':16,
            'discount': 0.99,
            'lr':0.02
        }
        self.hp.update(hp)
        self.device = torch.device("cpu")
        hidden_dim=128
        self.actor_critic = Descrete_PPO_actor_critic(state_dim, action_dim, [128,128])

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.hp['lr'], betas = (0.9, 0.999))
        self.optimizer.zero_grad()
        self.gs_num = 0
        self.completed_episodes = 0

        self.batch_episodes = []
        self.currently_building_episode = []
        self.name += "_%d_lr[%.4f]_b[%d]_CO-%s"%(hidden_dim, self.hp['lr'], self.hp['batch_size'], self.critic_objective)


    def get_action(self, state):
        action_probs, state_value = self.actor_critic(torch.from_numpy(state).float().unsqueeze(0))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)[0]
        action= action.item()
        return action, log_prob, state_value[0,0] # one value in

    def process_new_state(self, state):
        action_index, self.last_action_log_prob, self.last_state_value = self.get_action(state)
        return action_index

    def process_output(self, new_state, reward, is_finale_state):
        self.currently_building_episode += [(self.last_action_log_prob, self.last_state_value, reward)]
        if is_finale_state:
            self.batch_episodes += [self.currently_building_episode]
            self.currently_building_episode = []

        if len(self.batch_episodes) == self.hp['batch_size']:
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
                    qsa = rollout[t][2] + self.hp['discount'] * qsa
                    qsa_estimates.insert(0, qsa)
                qsa_estimates = torch.tensor(qsa_estimates)
                qsa_estimates = (qsa_estimates - qsa_estimates.mean()) / (qsa_estimates.std())
            elif self.critic_objective=="TD":
                with torch.no_grad():
                    torch_rollout = torch.tensor(rollout)
                    qsa_estimates = torch_rollout[:, 2]
                    qsa_estimates[:-1] += self.hp['discount']*torch_rollout[1:, 1]

            # rollout_torch = torch.tensor(rollout)
            # critic_obj = (qsa_estimates - rollout_torch[:,1])
            # actor_obj = -rollout_torch[:,0]*critic_obj.detach()/ len(self.batch_episodes)
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



