import os
from Agents.dnn_models import *
from utils import *
from Agents.GenericAgent import GenericAgent
from Agents.ICM import ICM
from torch.utils.data import DataLoader
from utils import NonSequentialDataset, safe_update_dict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPOParallel(GenericAgent):
    def __init__(self, state_dim, action_dim, hp, train=True):
        super(PPOParallel, self).__init__(train)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.train= train
        self.hp = {
            'concurrent_epsiodes':8,
            'horizon':128,
            'epochs': 3,
            'minibatch_size':32,
            'discount':0.99,
            'lr':0.01,
            'lr_decay':0.95,
            'epsilon_clip':0.2,
            'value_clip':0.5,
            'fe_layers':[64],
            'model_layers':[64],
            'entropy_weight':0.01,
            'grad_clip':0.5,
            'GAE': 1, # 1 for MC, 0 for TD,
            'curiosity_hp': None

        }
        safe_update_dict(self.hp, hp)

        if len(self.state_dim) > 1:
            feature_extractor = ConvNetFeatureExtracor(self.state_dim[0], self.hp['fe_layers'])
        else:
            feature_extractor = LinearFeatureExtracor(self.state_dim[0], self.hp['fe_layers'], batch_normalization=False,  activation=nn.ReLU())

        if type(self.action_dim) == list:
            self.num_outputs = len(self.action_dim[0])
            self.policy = ActorCriticModel(feature_extractor, self.num_outputs, self.hp['model_layers'], discrete=False, activation=nn.ReLU()).to(device)
        else:
            self.num_outputs=1
            self.policy = ActorCriticModel(feature_extractor, self.action_dim, self.hp['model_layers'], discrete=True, activation=nn.ReLU()).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hp['lr'])
        self.optimizer.zero_grad()

        if self.hp['curiosity_hp'] is None:
            self.curiosity = None
        else:
            self.curiosity = ICM(self.state_dim[0], self.action_dim, **self.hp['curiosity_hp'])

        self.name = 'PPO-Parallel'
        if self.hp['curiosity_hp'] is not None:
            self.name += "-ICM"
        self.name += "_lr[%.5f]_b[%d]_GAE[%.2f]_ec[%.1f]_l-%s-%s"%(
            self.hp['lr'], self.hp['concurrent_epsiodes'], self.hp['GAE'], self.hp['epsilon_clip'], str(self.hp['fe_layers']), str(self.hp['model_layers']))
        if self.hp['value_clip'] is not None:
            self.name += "_vc[%.1f]"%self.hp['value_clip']
        if  self.hp['grad_clip'] is not None:
            self.name += "_gc[%.1f]"%self.hp['grad_clip']

        self.num_steps = 0
        self.learn_steps = 0
        if self.hp['curiosity_hp'] is not None:
            self.states_memory = torch.zeros((self.hp['concurrent_epsiodes'], self.hp['horizon']+1) + self.state_dim).to(device)
        else:
            self.states_memory = torch.zeros((self.hp['concurrent_epsiodes'], self.hp['horizon']) + self.state_dim).to(device)
        self.actions_memory = torch.zeros((self.hp['concurrent_epsiodes'], self.hp['horizon'], self.num_outputs)).to(device)
        self.values_memory = torch.zeros((self.hp['concurrent_epsiodes'], self.hp['horizon'] + 1, 1)).to(device)
        self.logprobs_memory = torch.zeros((self.hp['concurrent_epsiodes'], self.hp['horizon'], 1)).to(device)
        self.rewards_memory = torch.zeros((self.hp['concurrent_epsiodes'], self.hp['horizon'], 1)).to(device)
        self.is_terminals_memory = torch.full((self.hp['concurrent_epsiodes'], self.hp['horizon'], 1), False).to(device)

    def load_state(self, path):
        if os.path.exists(path):
            # if trained on gpu but test on cpu use:
            self.policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def save_state(self, path):
        torch.save(self.policy.state_dict(), path)

    def evaluate_policy_on_state(self, state):
        torch_states = torch.from_numpy(state).to(device).float()
        torch_states = torch_states.unsqueeze(0)
        dists, _ = self.policy(torch_states)
        actions = dists.sample().detach()
        ourput_action = self._get_output_actions(actions)
        return ourput_action

    def process_states(self, states):
        torch_states = torch.from_numpy(states).to(device).float()
        dists, values = self.policy(torch_states)
        if self.num_steps == self.hp['horizon']:
            self.values_memory[:, self.num_steps] = values.detach()
            if self.hp['curiosity_hp'] is not None:
                self.states_memory[:, self.num_steps] = torch_states
            self._learn()
            self.learn_steps += 1

            if (self.learn_steps+1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.hp['lr_decay']
                self.reporter.add_costume_log("lr", self.learn_steps, self.optimizer.param_groups[0]['lr'])

            self.num_steps = 0

        actions = dists.sample().detach()
        self.states_memory[:,self.num_steps] = torch_states
        self.actions_memory[:, self.num_steps] = actions.view(-1,1)
        self.values_memory[:, self.num_steps] = values.detach()
        self.logprobs_memory[:, self.num_steps] = dists.log_prob(actions).detach().view(-1,1)

        return self._get_output_actions(actions)

    def update_step_results(self, next_states, rewards, is_next_state_terminals):
        self.rewards_memory[:, self.num_steps] = torch.from_numpy(rewards).to(device).view(-1, 1)
        self.is_terminals_memory[:, self.num_steps] = torch.from_numpy(is_next_state_terminals).to(device).view(-1, 1)
        self.num_steps += 1


    def _get_output_actions(self, actions):
        output_actions = actions.detach().cpu().numpy() # Using only this is problematic for super mario since it returns a 0-size np array in discrete action space
        if type(self.action_dim) == list:
            output_actions = np.clip(output_actions, self.action_dim[0], self.action_dim[1])

        return output_actions

    def _create_lerning_data(self):
        if self.curiosity is not None:
            cur_states = self.states_memory[:, :-1]
            next_states = self.states_memory[:, 1:]
            raw_rewards = 0
            intrinsic_reward = self.curiosity.get_intrinsic_reward(cur_states, next_states, self.actions_memory)
            self.reporter.add_costume_log("extrinsic_reward", self.num_actions, raw_rewards.mean())
            raw_rewards[:-1] += intrinsic_reward
            self.reporter.add_costume_log("curiosity_loss", self.num_actions, self.curiosity.get_last_debug_loss())
            self.reporter.add_costume_log("intrinsic_reward", self.num_actions, intrinsic_reward.mean())


        cur_values = self.values_memory[:,:-1]
        next_values = self.values_memory[:,1:]
        deltas = self.rewards_memory + self.hp['discount'] * next_values * (1 - self.is_terminals_memory) - cur_values
        advantages = monte_carlo_reward_batch(deltas, self.is_terminals_memory, self.hp['GAE'] * self.hp['discount'], device)
        rewards = advantages + cur_values
        advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-6)

        # Create a dataset from flatten data
        dataset = NonSequentialDataset(self.states_memory.view(-1, *self.states_memory.shape[2:]),
                                       cur_values.reshape(-1, *cur_values.shape[2:]), # view not working here (reshape copeies)
                                       self.actions_memory.view(-1, *self.actions_memory.shape[2:]),
                                       self.logprobs_memory.view(-1, *self.logprobs_memory.shape[2:]),
                                       rewards.view(-1, *rewards.shape[2:]),
                                       advantages.view(-1, *advantages.shape[2:]))
        dataloader = DataLoader(dataset, batch_size=self.hp['minibatch_size'], shuffle=True)

        return dataloader

    def _learn(self):
        dataloader = self._create_lerning_data()

        for _ in range(self.hp['epochs']):
            for (states_batch, old_policy_values_batch, old_policy_actions_batch, old_policy_loggprobs_batch, rewards_batch, advantages_batch) in dataloader:
                # Evaluating old actions and values with the target policy:
                dists, values = self.policy(states_batch)
                exploration_loss = -self.hp['entropy_weight'] * dists.entropy()
                value_loss = 0.5 * (values - rewards_batch).pow(2)
                if self.hp['value_clip'] is not None:
                    clipped_values = old_policy_values_batch + (values - old_policy_values_batch).clamp(
                        -self.hp['value_clip'], -self.hp['value_clip'])
                    clipepd_value_loss = 0.5 * (clipped_values - rewards_batch).pow(2)
                    critic_loss = torch.max(value_loss, clipepd_value_loss).mean()
                else:
                    critic_loss = value_loss

                # Finding the ratio (pi_theta / pi_theta_old):
                logprobs = dists.log_prob(old_policy_actions_batch.view(-1))
                ratios = torch.exp(logprobs.view(-1,1) - old_policy_loggprobs_batch)
                # Finding Surrogate actor Loss:
                ratios = torch.clamp(ratios, 0, 10)  # TODO temporal experiment
                surr1 = advantages_batch * ratios
                surr2 = advantages_batch * torch.clamp(ratios, 1 - self.hp['epsilon_clip'],
                                                       1 + self.hp['epsilon_clip'])
                actor_loss = -torch.min(surr1, surr2)

                loss = actor_loss.mean() + critic_loss.mean() + exploration_loss.mean()
                loss.backward()

                if self.hp['grad_clip'] is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.hp['grad_clip'])
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.reporter.add_costume_log("actor_loss", None, actor_loss.mean().item())
                self.reporter.add_costume_log("critic_loss", None, critic_loss.mean().item())
                self.reporter.add_costume_log("dist_entropy", None, -exploration_loss.mean().item())
                self.reporter.add_costume_log("ratios", None, ratios.mean().item())
                self.reporter.add_costume_log("values", None, values.mean().item())

