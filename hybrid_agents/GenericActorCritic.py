import os
from dnn_models import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.state_values = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def __len__(self):
        return len(self.state_values)

    def add_sample(self, value, action_log_prob, reward, is_terminals):
        self.state_values += [value]
        self.logprobs += [action_log_prob]
        self.rewards += [reward]
        self.is_terminals += [is_terminals]

    def get_as_tensors(self, device):
        states = torch.stack(self.state_values).to(device)
        logprobs = torch.stack(self.logprobs).to(device)
        return states, logprobs, self.rewards, self.is_terminals

    def clear_memory(self):
        del self.state_values[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]



class ActorCritic(object):
    def __init__(self, state_dim, action_dim, hp, train=True):
        self.name = 'ActorCritic'
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.train= train
        self.hp = {
            'batch_size':4000,
            'discount':0.99,
            'lr':0.01,
            'lr_decay':0.95,
            'hidden_layer_size':128
        }
        self.hp.update(hp)
        self.samples = Memory()

        if type(self.state_dim) == tuple:
            feature_extractor = ConvNetFeatureExtracor(self.state_dim[0])
        else:
            feature_extractor = LinearFeatureExtracor(self.state_dim, self.hp['hidden_layer_size'])
        if type(self.action_dim) == list:
            self.policy = ContinousActorCriticModdel(feature_extractor, len(self.action_dim), self.hp['hidden_layer_size']).to(device)
        else:
            self.policy = DiscreteActorCriticModel(feature_extractor, self.action_dim, self.hp['hidden_layer_size']).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hp['lr'])
        self.optimizer.zero_grad()
        self.MseLoss = nn.MSELoss()
        self.learn_steps = 0

        self.name += "_lr[%.4f]_b[%d]"%(self.hp['lr'], self.hp['batch_size'])

    def process_new_state(self, state):
        state = torch.from_numpy(np.array(state)).to(device).float()
        dist, value = self.policy(state.unsqueeze(0))

        action = dist.sample()

        self.last_state_value = value[0,0]
        self.last_action_log_prob = dist.log_prob(action)[0]

        action = action.detach().cpu().numpy()[0]
        return action

    def process_output(self, new_state, reward, is_finale_state):
        self.samples.add_sample(self.last_state_value, self.last_action_log_prob, reward, is_finale_state)

        if len(self.samples) == self.hp['batch_size']:
            self._learn()
            self.samples.clear_memory()
            self.learn_steps += 1

            if (self.learn_steps+1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.hp['lr_decay']

    def _learn(self):
        state_values, logprobs, raw_rewards, is_terminals = self.samples.get_as_tensors(device)

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

        # for _ in range(self.hp['epochs']):
        advantage = rewards.detach() - state_values.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
        actor_loss = -logprobs*advantage
        critic_loss = (0.5*(state_values - rewards).pow(2))
        # critic_loss = torch.nn.functional.smooth_l1_loss(state_values, rewards)
        loss = (actor_loss + critic_loss).mean()
        # take gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def load_state(self, path):
        if os.path.exists(path):
            # self.policy_old.load_state_dict(torch.load(path))
            # if trained on gpu but test on cpu use:
            self.policy_old.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def save_state(self, path):
        torch.save(self.policy.state_dict(), path)

    def get_stats(self):
        return "Gs: %d; LR: %.5f"%(self.learn_steps, self.optimizer.param_groups[0]['lr'])



