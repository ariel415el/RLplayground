import os
from Agents.dnn_models import *
from utils import *
from Agents.GenericAgent import GenericAgent
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


class ActorCritic(GenericAgent):
    def __init__(self, state_dim, action_dim, hp, train=True):
        super(ActorCritic, self).__init__( train)
        self.name = 'ActorCritic'
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.train= train
        self.hp = {
            'batch_episodes':3,
            'discount':0.99,
            'lr':0.01,
            'lr_decay':0.95,
            'hidden_layers':[128,128],
            "GAE": 1
        }
        self.hp.update(hp)
        self.samples = Memory()

        if len(self.state_dim) > 1:
            feature_extractor = ConvNetFeatureExtracor(self.state_dim[0])
        else:
            feature_extractor = LinearFeatureExtracor(self.state_dim[0], self.hp['hidden_layers'][0])

        if type(self.action_dim) == list:
            self.policy = ActorCriticModel(feature_extractor, len(self.action_dim[0]), self.hp['hidden_layers'], discrete=False).to(device)
        else:
            self.policy = ActorCriticModel(feature_extractor, self.action_dim[0], self.hp['hidden_layers'][1:], discrete=True).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hp['lr'])
        self.optimizer.zero_grad()
        self.learn_steps = 0
        self.episodes_in_cur_batch = 0

        self.name += "_lr[%.5f]_b[%d]_GAE[%.2f]_l-%s"%(self.hp['lr'], self.hp['batch_episodes'], self.hp['GAE'],self.hp['hidden_layers'])

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

        if is_finale_state:
            self.episodes_in_cur_batch += 1
        if self.episodes_in_cur_batch == self.hp['batch_episodes']:
            self._learn()
            self.samples.clear_memory()
            self.learn_steps += 1
            self.episodes_in_cur_batch = 0

            if (self.learn_steps+1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.hp['lr_decay']

    def _learn(self):
        state_values, logprobs, raw_rewards, is_terminals = self.samples.get_as_tensors(device)

        rewards = monte_carlo_reward(raw_rewards, is_terminals, self.hp['discount'], device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        advantage = GenerelizedAdvantageEstimate(self.hp['GAE'], state_values, raw_rewards, is_terminals, self.hp['discount'], device).detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

        actor_loss = -logprobs*advantage
        critic_loss = 0.5*(state_values - rewards).pow(2)
        loss = (actor_loss + critic_loss).mean()
        # take gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.reporter.update_agent_stats("actor_loss", self.learn_steps, actor_loss.mean().item())
        self.reporter.update_agent_stats("critic_loss", self.learn_steps, critic_loss.mean().item())

    def load_state(self, path):
        if os.path.exists(path):
            # self.policy_old.load_state_dict(torch.load(path))
            # if trained on gpu but test on cpu use:
            self.policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def save_state(self, path):
        torch.save(self.policy.state_dict(), path)

    def get_stats(self):
        return "Gs: %d; LR: %.5f"%(self.learn_steps, self.optimizer.param_groups[0]['lr'])



