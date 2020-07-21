##############################################
### Credits to nikhilbarhate99/PPO-PyTorch ###
##############################################
import os
from Agents.dnn_models import *
from utils.utils import *
from Agents.GenericAgent import GenericAgent
from Agents.ICM import ICM
from torch.utils.data import DataLoader
from utils.utils import BasicDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Memory:
    def __init__(self):
        self.states = []
        self.state_values = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_ns_terminals = []

    def __len__(self):
        return len(self.states)

    def add_sample(self, state, value, action, action_log_prob, reward, is_ns_terminal):
        self.states += [state]
        self.state_values += [value]
        self.actions += [action]
        self.logprobs += [action_log_prob]
        self.rewards += [reward]
        self.is_ns_terminals += [is_ns_terminal]

    def get_as_tensors(self, device):
        states = torch.stack(self.states).to(device)
        values = torch.tensor(self.state_values).to(device)
        actions = torch.tensor(self.actions).to(device)
        logprobs = torch.tensor(self.logprobs).to(device)
        return states, values, actions, logprobs, self.rewards, self.is_ns_terminals

    def clear_memory(self):
        self.states = []
        self.state_values = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_ns_terminals = []


class PPO(GenericAgent):
    def __init__(self, state_dim, action_dim, hp, train=True):
        super(PPO, self).__init__(train)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.train= train
        self.hp = {
            'batch_episodes':3,
            'horizon':None,
            'epochs': 4,
            'minibatch_size':32,
            'discount':0.99,
            'lr':0.01,
            'lr_decay':0.95,
            'epsilon_clip':0.2,
            'value_clip':0.5,
            'fe_layers':[64],
            'features_bn':False,
            'model_layers':[64],
            'entropy_weight':0.01,
            'grad_clip':0.5,
            'GAE': 1, # 1 for MC, 0 for TD,
            'curiosity_hp': None

        }
        safe_update_dict(self.hp, hp)
        self.samples = Memory()
        if len(self.state_dim) > 1:
            feature_extractor = ConvNetFeatureExtracor(self.state_dim[0], self.hp['fe_layers'])
        else:
            feature_extractor = LinearFeatureExtracor(self.state_dim[0], self.hp['fe_layers'], batch_normalization=self.hp['features_bn'],  activation=nn.ReLU())

        if type(self.action_dim) == list:
            self.policy = ActorCriticModel(feature_extractor, len(self.action_dim[0]), self.hp['model_layers'], discrete=False, activation=nn.ReLU()).to(device)
        else:
            self.policy = ActorCriticModel(feature_extractor, self.action_dim, self.hp['model_layers'], discrete=True, activation=nn.ReLU()).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hp['lr'])
        self.optimizer.zero_grad()
        self.learn_steps = 0
        self.num_actions = 0
        self.episodes_in_cur_batch = 0

        # Set agent name
        if self.hp['curiosity_hp'] is None:
            self.curiosity = None
        else:
            self.curiosity = ICM(self.state_dim[0], self.action_dim, **self.hp['curiosity_hp'])
            from sklearn.preprocessing import StandardScaler
            self.reward_scaler = RunningStats((1,), computation_module=np)
            self.state_scaler = RunningStats(self.state_dim, computation_module=torch)

        self.name = 'PPO'
        if self.hp['curiosity_hp'] is not None:
            self.name += "-ICM"
        self.name += "_lr[%.5f]_b[%d]_GAE[%.2f]_ec[%.1f]"%(self.hp['lr'], self.hp['batch_episodes'], self.hp['GAE'], self.hp['epsilon_clip'])
        self.name += "_fl-%s_ml-%s"%(str(self.hp['fe_layers']), str(self.hp['model_layers']))
        if self.hp['value_clip'] is not None:
            self.name += "_vc[%.1f]"%self.hp['value_clip']
        if  self.hp['grad_clip'] is not None:
            self.name += "_gc[%.1f]"%self.hp['grad_clip']

    def process_new_state(self, state):
        state = torch.from_numpy(np.array(state)).to(device).float()
        dist, value = self.policy(state.unsqueeze(0))
        # dist, value = self.policy(torch.cat([state.unsqueeze(0), state.unsqueeze(0)],axis=0))

        action = dist.sample()[0]

        self.last_action_log_prob = dist.log_prob(action)[0].item()
        self.last_state = state
        self.last_value = value[0,0].item() # no need gradient
        if type(self.action_dim) == list:
            action = action.detach().cpu().numpy() # Using only this is problematic for super mario since it returns a 0-size np array in discrete action space
            output_action = np.clip(action, self.action_dim[0], self.action_dim[1])
        else:
            action = output_action = action.item()
        self.last_action = action

        self.num_actions += 1
        return output_action

    def process_output(self, next_state, reward, is_finale_state):
        self.samples.add_sample(self.last_state, self.last_value, self.last_action, self.last_action_log_prob, reward, is_finale_state)
        if is_finale_state:
            self.episodes_in_cur_batch += 1
        if self.episodes_in_cur_batch == self.hp['batch_episodes']:
            self._learn(next_state)
            self.samples.clear_memory()
            self.learn_steps += 1
            self.episodes_in_cur_batch = 0

            if (self.learn_steps+1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.hp['lr_decay']
                self.reporter.add_costume_log("lr", self.learn_steps, self.optimizer.param_groups[0]['lr'])

    def _learn(self, next_state):
        states, old_policy_values, old_policy_actions, old_policy_loggprobs, raw_rewards, is_next_state_terminals = self.samples.get_as_tensors(device)
        raw_rewards = np.array(raw_rewards)

        if self.curiosity is not None:
            combine_factor = 0.01
            self.reporter.add_costume_log("extrinsic_reward", self.num_actions, raw_rewards.mean())
            self.reward_scaler.update(raw_rewards)
            self.state_scaler.update(states)
            raw_rewards = self.reward_scaler.scale(raw_rewards)
            states = self.state_scaler.scale(states).float()
            next_state = torch.from_numpy(next_state[None]).to(device).float()
            next_states = torch.cat((states[1:], next_state), axis=0)
            intrinsic_reward = self.curiosity.get_intrinsic_reward(states, next_states, old_policy_actions)
            raw_rewards += combine_factor*(intrinsic_reward - raw_rewards)
            self.reporter.add_costume_log("curiosity_loss", self.num_actions, self.curiosity.get_last_debug_loss())
            self.reporter.add_costume_log("intrinsic_reward", self.num_actions, intrinsic_reward.mean())

        advantages, rewards = GenerelizedAdvantageEstimate(self.hp['GAE'], old_policy_values, raw_rewards, is_next_state_terminals, self.hp['discount'], device, horizon=self.hp['horizon'])
        advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-6)

        # Optimize policy for K epochs:
        dataset = BasicDataset(states, old_policy_values, old_policy_actions, old_policy_loggprobs, rewards, advantages)
        dataloader = DataLoader(dataset, batch_size=self.hp['minibatch_size'], shuffle=True)
        for _ in range(self.hp['epochs']):
            for (states_batch, old_policy_values_batch, old_policy_actions_batch, old_policy_loggprobs_batch, rewards_batch, advantages_batch) in dataloader:
                # Evaluating old actions and values with the target policy:
                dists, values = self.policy(states_batch)
                values = values.view(-1)
                exploration_loss = -self.hp['entropy_weight'] * dists.entropy()
                value_loss = 0.5 * (values - rewards_batch).pow(2)
                if self.hp['value_clip'] is not None:
                    clipped_values = old_policy_values_batch + (values - old_policy_values_batch).clamp(-self.hp['value_clip'], -self.hp['value_clip'])
                    clipepd_value_loss = 0.5*(clipped_values - rewards_batch).pow(2)
                    critic_loss = torch.max(value_loss, clipepd_value_loss).mean()
                else:
                    critic_loss = value_loss

                # Finding the ratio (pi_theta / pi_theta_old):
                logprobs = dists.log_prob(old_policy_actions_batch)
                ratios = torch.exp(logprobs.view(-1) - old_policy_loggprobs_batch)
                # Finding Surrogate actor Loss:
                ratios = torch.clamp(ratios,0,10) # TODO temporal experiment
                surr1 = advantages_batch* ratios
                surr2 = advantages_batch* torch.clamp(ratios, 1 - self.hp['epsilon_clip'], 1 + self.hp['epsilon_clip'])
                actor_loss = -torch.min(surr1, surr2)

                loss = actor_loss.mean() + critic_loss.mean() + exploration_loss.mean()
                loss.backward()
                # loss = actor_loss + critic_loss + exploration_loss
                # loss.mean().backward()
                if self.hp['grad_clip'] is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.hp['grad_clip'])
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.reporter.add_costume_log("actor_loss", None, actor_loss.mean().item())
                self.reporter.add_costume_log("critic_loss", None, critic_loss.mean().item())
                self.reporter.add_costume_log("dist_entropy", None, -exploration_loss.mean().item())
                self.reporter.add_costume_log("ratios", None, ratios.mean().item())
                self.reporter.add_costume_log("values", None, values.mean().item())
                # self.reporter.add_histogram("actions", old_policy_actions_batch.cpu().numpy().reshape(-1))

    def load_state(self, path):
        if os.path.exists(path):
            # if trained on gpu but test on cpu use:
            self.policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def save_state(self, path):
        torch.save(self.policy.state_dict(), path)
