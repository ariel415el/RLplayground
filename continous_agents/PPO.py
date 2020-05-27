##############################################
### Credits to nikhilbarhate99/PPO-PyTorch ###
##############################################
import os
import torch
from torch import nn
import torch.distributions as D
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOContinousActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, layers_dims):
        super(PPOContinousActorCritic, self).__init__()
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

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def __len__(self):
        return len(self.states)

    def add_sample(self, state, action, action_log_prob, reward, is_terminals):
        self.actions += [action]
        self.states += [state]
        self.logprobs += [action_log_prob]
        self.rewards += [reward]
        self.is_terminals += [is_terminals]

    def get_as_tensors(self, device):
        states = torch.stack(self.states).to(device)
        actions = torch.stack(self.actions).to(device)
        logprobs = torch.tensor(self.logprobs).to(device)
        return states, actions, logprobs, self.rewards, self.is_terminals

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class PPO_continous_action(object):
    def __init__(self, state_dim, action_bounderies, max_episodes, train=True):
        self.name = 'PPO'
        self.state_dim = state_dim
        self.action_bounderies = action_bounderies
        self.action_dim = len(action_bounderies[0])
        self.max_episodes = max_episodes
        self.train= train
        self.batch_size = 4000
        self.discount = 0.99
        self.action_std = 0.5
        self.action_var = torch.full((self.action_dim,), self.action_std**2)

        self.samples = Memory()
        self.steps_per_iteration=80
        self.epsilon_clip = 0.2

        self.lr = 0.0003
        self.lr_decay = 0.995
        layers = [150, 120]
        self.policy = PPOContinousActorCritic(state_dim, self.action_dim, layers).to(device)
        self.policy_old = PPOContinousActorCritic(state_dim, self.action_dim, layers).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer.zero_grad()
        self.MseLoss = nn.MSELoss()
        self.learn_steps = 0
        self.completed_episodes = 0

        self.name += "_lr[%.4f]_b[%d]"%(self.lr, self.batch_size)

    def get_action_dist(self, model,  state_tensor):
        mus = model.get_mu(state_tensor)
        cov_mat = torch.diag(self.action_var).to(device)
        dist = D.MultivariateNormal(mus, cov_mat)

        return dist

    def process_new_state(self, state):
        state = torch.from_numpy(state).to(device).float()
        dist = self.get_action_dist(self.policy_old, state)

        action = dist.sample()

        self.last_state = state
        self.last_action = action
        self.last_action_log_prob = dist.log_prob(action)

        action = action.detach().cpu().numpy()
        return action
        # return np.clip(action, self.action_bounderies[0], self.action_bounderies[0])

    def process_output(self, new_state, reward, is_finale_state):
        self.samples.add_sample(self.last_state , self.last_action, self.last_action_log_prob, reward, is_finale_state)

        if len(self.samples) == self.batch_size:
            self._learn()
            self.samples.clear_memory()
            self.learn_steps += 1

            if (self.learn_steps+1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.lr_decay


    def _learn(self):
        old_states, old_actions, old_logprobs, raw_rewards, is_terminals = self.samples.get_as_tensors(device)

        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(raw_rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.discount * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Optimize policy for K epochs:
        for _ in range(self.steps_per_iteration):
            # Evaluating old actions and values :
            dists = self.get_action_dist(self.policy, old_states)
            logprobs = dists.log_prob(old_actions)
            dist_entropies = dists.entropy()
            state_values = self.policy.get_value(old_states).view(-1)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.float(), rewards.float()) - 0.01 * dist_entropies
            loss = loss.double()
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def load_state(self, path):
        if os.path.exists(path):
            # self.policy_old.load_state_dict(torch.load(path))
            # if trained on gpu but test on cpu use:
            self.policy_old.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def save_state(self, path):
        torch.save(self.policy.state_dict(), path)

    def get_stats(self):
        return "Gs: %d; LR: %.5f"%(self.learn_steps, self.optimizer.param_groups[0]['lr'])



