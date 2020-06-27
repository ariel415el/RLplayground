##############################################
### Credits to nikhilbarhate99/PPO-PyTorch ###
##############################################
import os
from dnn_models import *
from utils import *
from GenericAgent import GenericAgent
from ICM import ICM
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.states = []
        self.state_values = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def __len__(self):
        return len(self.states)

    def add_sample(self, state, value, action, action_log_prob, reward, is_terminals):
        self.states += [state]
        self.state_values += [value]
        self.actions += [action]
        self.logprobs += [action_log_prob]
        self.rewards += [reward]
        self.is_terminals += [is_terminals]

    def get_as_tensors(self, device):
        states = torch.stack(self.states).to(device)
        values = torch.tensor(self.state_values).to(device)
        actions = torch.stack(self.actions).to(device)
        logprobs = torch.stack(self.logprobs).to(device)
        return states, values, actions, logprobs, self.rewards, self.is_terminals

    def clear_memory(self):
        del self.states[:]
        del self.state_values[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class temp_actor_critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.policy_out = nn.Linear(64,action_dim)
        self.value_out = nn.Linear(64, 1)
        self.softmax = torch.nn.Softmax(dim=1)
    def get_dist(self, features):
        probs = self.policy_out(features)
        probs = self.softmax(probs)
        dist = D.Categorical(probs)
        return dist

    def forward(self, state):
        x = self.input(state)
        dist = self.get_dist(x)
        value = self.value_out(x)
        return dist, value

class HybridPPO_ICM(GenericAgent):
    def __init__(self, state_dim, action_dim, hp, train=True):
        super(HybridPPO_ICM, self).__init__(train)
        self.name = 'PPO_ICM'
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.train= train
        self.hp = {
            'batch_episodes':3,
            'minibatch_size':32,
            'epochs': 4,
            'discount':0.99,
            'lr':0.01,
            'lr_decay':0.95,
            'epsiolon_clip':0.2,
            'value_clip':0.5,
            'hidden_layers':[128,128],
            'entropy_weight':0.01,
            'grad_clip':0.5,
            'GAE': 1, # 1 for MC, 0 for TD
            'use_extrinsic_reward':True,
            'intrinsic_reward_scale':1.0,
            'curiosity_hidden_dim':128,
            'curiosity_lr':0.001

        }
        self.hp.update(hp)
        self.samples = Memory()
        if type(self.state_dim) == tuple:
            feature_extractor = ConvNetFeatureExtracor(self.state_dim[0])
        else:
            feature_extractor = LinearFeatureExtracor(self.state_dim, self.hp['hidden_layers'][0], activation=torch.tanh)

        if type(self.action_dim) == list:
            self.policy = ActorCriticModel(feature_extractor, len(self.action_dim[0]), self.hp['hidden_layers'], discrete=False).to(device)
        else:
            # self.policy = ActorCriticModel(feature_extractor, self.action_dim, self.hp['hidden_layers'][1:], discrete=True).to(device)
            self.policy = temp_actor_critic(self.state_dim, self.action_dim).to(device)

        self.curiosity = ICM(self.state_dim, self.action_dim, self.hp['curiosity_hidden_dim'], lr=self.hp['curiosity_lr'], intrinsic_reward_scale=self.hp['intrinsic_reward_scale'])

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hp['lr'])
        self.optimizer.zero_grad()
        self.learn_steps = 0
        self.num_actions = 0
        self.episodes_in_cur_batch = 0

        self.name += "_lr[%.4f]_b[%d]_GAE[%.2f]"%(self.hp['lr'], self.hp['batch_episodes'], self.hp['GAE'])
        if self.hp['value_clip'] is not None:
            self.name += "_vc[%.1f]"%self.hp['value_clip']
        if  self.hp['grad_clip'] is not None:
            self.name += "_gc[%.1f]"%self.hp['grad_clip']
        if not self.hp['use_extrinsic_reward']:
            self.name += '_no-ER'

    def process_new_state(self, state):
        state = torch.from_numpy(np.array(state)).to(device).float()
        dist, value = self.policy(state.unsqueeze(0))

        action = dist.sample()[0]

        self.last_state = state
        self.last_value = value[0,0].item() # no need gradient
        self.last_action = action
        self.last_action_log_prob = dist.log_prob(action)[0]

        action = action.detach().cpu().numpy()
        if type(self.action_dim) == list:
            action = np.clip(action, self.action_dim[0], self.action_dim[1])
        self.num_actions += 1
        return action

    def process_output(self, new_state, reward, is_finale_state):
        self.samples.add_sample(self.last_state, self.last_value, self.last_action, self.last_action_log_prob, reward, is_finale_state)
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
                self.reporter.update_agent_stats("lr", self.learn_steps, self.optimizer.param_groups[0]['lr'])

    def _learn(self):
        old_states, old_values, old_actions, old_logprobs, raw_rewards, is_terminals = self.samples.get_as_tensors(device)

        intrinsic_reward = self.curiosity.get_intrinsic_loss(old_states[:-1], old_states[1:], old_actions[:-1])
        self.reporter.update_agent_stats("curiosity_loss", self.num_actions, self.curiosity.get_last_debug_loss())

        total_r = 0
        scores = []
        for r,t in zip(intrinsic_reward, is_terminals):
            total_r += r
            if t :
                scores += [total_r]
                total_r = 0
        scores += [total_r]
        self.reporter.update_agent_stats("total_initrisic_rewards", self.num_actions, np.mean(scores))

        raw_rewards = np.array(raw_rewards)
        if self.hp['use_extrinsic_reward']:
            raw_rewards[:-1] += intrinsic_reward.numpy()
        else:
            raw_rewards[:-1] = intrinsic_reward.numpy()
        rewards = monte_carlo_reward(raw_rewards, is_terminals, self.hp['discount'], device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        advantages = GenerelizedAdvantageEstimate(self.hp['GAE'], old_values, raw_rewards, is_terminals, self.hp['discount'], device).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        debug_actor_loss = []
        debug_critic_loss = []
        debug_entropy_loss = []
        debug_total_loss= []
        # Optimize policy for K epochs:
        for _ in range(self.hp['epochs']):
            # Evaluating old actions and values with the target policy:
            dists, values = self.policy(old_states)

            exploration_loss = -self.hp['entropy_weight'] * dists.entropy()

            values = values.view(-1)
            value_loss = 0.5 * (values - rewards).pow(2)
            if self.hp['value_clip'] is not None:
                clipped_values = old_values + (values - old_values).clamp(-self.hp['value_clip'], -self.hp['value_clip'])
                clipepd_value_loss = 0.5*(clipped_values - rewards).pow(2)
                critic_loss = torch.min(value_loss, clipepd_value_loss).mean()
            else:
                critic_loss = value_loss

            # Finding the ratio (pi_theta / pi_theta__old):
            logprobs = dists.log_prob(old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate actor Loss:
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.hp['epsiolon_clip'], 1 + self.hp['epsiolon_clip']) * advantages
            actor_loss = -torch.min(surr1, surr2)

            loss = actor_loss + critic_loss + exploration_loss
            self.optimizer.zero_grad()
            loss.mean().backward()
            if self.hp['grad_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.hp['grad_clip'])
            self.optimizer.step()
            debug_actor_loss += [actor_loss.mean().item()]
            debug_critic_loss += [critic_loss.mean().item()]
            debug_entropy_loss += [exploration_loss.mean().item()]
            debug_total_loss += [loss.mean().item()]

        self.reporter.update_agent_stats("actor_loss", self.num_actions, np.mean(debug_actor_loss))
        self.reporter.update_agent_stats("critic_loss", self.num_actions, np.mean(debug_critic_loss))
        self.reporter.update_agent_stats("dist_entropy", self.num_actions, -np.mean(debug_entropy_loss))
        self.reporter.update_agent_stats("total_loss", self.num_actions, np.mean(debug_total_loss))

    def load_state(self, path):
        if os.path.exists(path):
            # self.policy_old.load_state_dict(torch.load(path))
            # if trained on gpu but test on cpu use:
            self.policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def save_state(self, path):
        torch.save(self.policy.state_dict(), path)

    def get_stats(self):
        return "Gs: %d; LR: %.5f"%(self.learn_steps, self.optimizer.param_groups[0]['lr'])



