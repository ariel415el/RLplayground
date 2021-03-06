import os
from Agents.dnn_models import *
from utils.utils import *
from Agents.GenericAgent import GenericAgent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def __len__(self):
        return len(self.logprobs)

    def add_sample(self, action_log_prob, reward, is_terminals):
        self.logprobs += [action_log_prob]
        self.rewards += [reward]
        self.is_terminals += [is_terminals]

    def get_as_tensors(self, device):
        logprobs = torch.stack(self.logprobs).to(device)
        return logprobs, self.rewards, self.is_terminals

    def clear_memory(self):
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class VanilaPolicyGradient(GenericAgent):
    def __init__(self, state_dim, action_dim, hp, train=True):
        super(VanilaPolicyGradient, self).__init__(train)
        self.name = 'Vanila-PG'
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.train= train
        self.hp = {
            "batch_episodes": 3,
            'discount':0.99,
            'lr':0.01,
            'lr_decay':0.95,
            'hidden_layers':[128,128]
        }
        self.hp.update(hp)
        self.samples = Memory()
        self.reporter = None
        if len(self.state_dim) > 1:
            feature_extractor = ConvNetFeatureExtracor(self.state_dim[0])
        else:
            feature_extractor = LinearFeatureExtracor(self.state_dim[0], self.hp['hidden_layers'][0])
        if type(self.action_dim) == list:
            self.policy = CountinousActor(feature_extractor, len(self.action_dim[0]), self.hp['hidden_layers']).to(device)
        else:
            self.policy = DiscreteActor(feature_extractor, self.action_dim, self.hp['hidden_layers'][1:]).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hp['lr'])
        self.optimizer.zero_grad()
        self.learn_steps = 0
        self.episodes_in_cur_batch = 0

        self.name += "_lr[%.4f]_b[%d]_l-%s"%(self.hp['lr'], self.hp['batch_episodes'],self.hp['hidden_layers'])


    def process_new_state(self, state):
        state = torch.from_numpy(np.array(state)).to(device).float()
        dist = self.policy(state.unsqueeze(0))

        action = dist.sample()

        self.last_action_log_prob = dist.log_prob(action)[0]

        action = action.detach().cpu().numpy()[0]
        return action

    def process_output(self, new_state, reward, is_finale_state):
        self.samples.add_sample(self.last_action_log_prob, reward, is_finale_state)
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
        logprobs, raw_rewards, is_terminals = self.samples.get_as_tensors(device)
        raw_rewards = np.array(raw_rewards)
        raw_rewards = (raw_rewards - raw_rewards.mean()) / (raw_rewards.std() + 1e-5)
        # Monte Carlo estimate of rewards:
        rewards = discount(raw_rewards, is_terminals, self.hp['discount'], device)
        actor_loss = (-logprobs*rewards).mean()

        # take gradient step
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        self.reporter.update_agent_stats("loss", self.learn_steps, actor_loss.item())

    def load_state(self, path):
        if os.path.exists(path):
            self.policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def save_state(self, path):
        torch.save(self.policy.state_dict(), path)

    def get_stats(self):
        return "Gs: %d; LR: %.5f"%(self.learn_steps, self.optimizer.param_groups[0]['lr'])




