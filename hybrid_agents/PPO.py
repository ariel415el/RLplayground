##############################################
### Credits to nikhilbarhate99/PPO-PyTorch ###
##############################################
import os
from dnn_models import *
import torch.distributions as D
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        logprobs = torch.stack(self.logprobs).to(device)
        return states, actions, logprobs, self.rewards, self.is_terminals

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class HybridPPO(object):
    def __init__(self, state_dim, action_dim, hp, train=True):
        self.name = 'PPO'
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.train= train
        self.hp = {
            'epoch_size':4000,
            'batch_size':4000,
            'discount':0.99,
            'lr':0.01,
            'lr_decay':0.995,
            'epsiolon_clip':0.2,
            'epochs':32,
            'hidden_layer_size':128,
            'entropy_weight':0.01
        }
        self.hp.update(hp)
        self.hp['batch_size'] = min(self.hp['epoch_size'], self.hp['batch_size'])
        self.samples = Memory()

        if type(self.state_dim) == tuple:
            feature_extractor = ConvNetFeatureExtracor(self.state_dim[0])
        else:
            feature_extractor = LinearFeatureExtracor(self.state_dim, self.hp['hidden_layer_size'], activation=torch.tanh)
        if type(self.action_dim) == list:
            self.policy = ContinousActorCriticModdel(feature_extractor, len(self.action_dim), self.hp['hidden_layer_size']).to(device)
        else:
            self.policy = DiscreteActorCriticModel(feature_extractor, self.action_dim, self.hp['hidden_layer_size']).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hp['lr'])
        self.optimizer.zero_grad()
        self.MseLoss = nn.MSELoss()
        self.learn_steps = 0
        self.completed_episodes = 0

        self.name += "_lr[%.4f]_b[%d]"%(self.hp['lr'], self.hp['batch_size'])

    def process_new_state(self, state):
        state = torch.from_numpy(np.array(state)).to(device).float()
        dist = self.policy.get_action_dist(state.unsqueeze(0))

        action = dist.sample()[0]

        self.last_state = state
        self.last_action = action
        self.last_action_log_prob = dist.log_prob(action)[0]

        action = action.detach().cpu().numpy()
        return action

    def process_output(self, new_state, reward, is_finale_state):
        self.samples.add_sample(self.last_state , self.last_action, self.last_action_log_prob, reward, is_finale_state)

        if len(self.samples) == self.hp['epoch_size']:
            self._learn()
            self.samples.clear_memory()
            self.learn_steps += 1

            if (self.learn_steps+1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.hp['lr_decay']

    def _learn(self):
        old_states, old_actions, old_logprobs, raw_rewards, is_terminals = self.samples.get_as_tensors(device)

        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(raw_rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.hp['discount'] * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Optimize policy for K epochs:
        for _ in range(self.hp['epochs']):
            # Evaluating old actions and values with the target policy:
            dists, values = self.policy(old_states)
            logprobs = dists.log_prob(old_actions)
            dist_entropies = dists.entropy()
            state_values = values.view(-1)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.hp['epsiolon_clip'], 1 + self.hp['epsiolon_clip']) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.float(), rewards.float()) - self.hp['entropy_weight'] * dist_entropies
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def load_state(self, path):
        if os.path.exists(path):
            # self.policy_old.load_state_dict(torch.load(path))
            # if trained on gpu but test on cpu use:
            self.policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def save_state(self, path):
        torch.save(self.policy.state_dict(), path)

    def get_stats(self):
        return "Gs: %d; LR: %.5f"%(self.learn_steps, self.optimizer.param_groups[0]['lr'])



