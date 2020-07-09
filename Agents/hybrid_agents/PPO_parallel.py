##############################################
### Credits to nikhilbarhate99/PPO-PyTorch ###
##############################################
import os
from Agents.dnn_models import *
from utils import *
from Agents.GenericAgent import GenericAgent
from Agents.ICM import ICM
from torch.utils.data import DataLoader
from utils import NonSequentialDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HybridPPOParallel(GenericAgent):
    def __init__(self, state_dim, action_dim, hp, curiosity=None, train=True):
        super(HybridPPOParallel, self).__init__(train)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.train= train
        self.hp = {
            'batch_episodes':3,
            'horizon':128,
            'epochs': 4,
            'minibatch_size':32,
            'discount':0.99,
            'lr':0.01,
            'lr_decay':0.95,
            'epsilon_clip':0.2,
            'value_clip':0.5,
            'features_layers':[64],
            'model_layers':[64],
            'entropy_weight':0.01,
            'grad_clip':0.5,
            'GAE': 1, # 1 for MC, 0 for TD,
            'curiosity_hp': None

        }
        self.hp.update(hp)

        if len(self.state_dim) > 1:
            feature_extractor = ConvNetFeatureExtracor(self.state_dim[0], self.hp['features_layers'][0])
        else:
            feature_extractor = LinearFeatureExtracor(self.state_dim[0], self.hp['features_layers'], batch_normalization=False,  activation=nn.ReLU())

        if type(self.action_dim) == list:
            self.policy = ActorCriticModel(feature_extractor, len(self.action_dim[0]), self.hp['model_layers'], discrete=False, activation=nn.ReLU()).to(device)
        else:
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
        self.name += "_lr[%.5f]_b[%d]_GAE[%.2f]_ec[%.1f]"%(self.hp['lr'], self.hp['batch_episodes'], self.hp['GAE'], self.hp['epsilon_clip'])
        if self.hp['value_clip'] is not None:
            self.name += "_vc[%.1f]"%self.hp['value_clip']
        if  self.hp['grad_clip'] is not None:
            self.name += "_gc[%.1f]"%self.hp['grad_clip']

    def process_single_state(self, state):
        state = torch.from_numpy(np.array(state)).to(device).float()
        dist, _ = self.policy(state.unsqueeze(0))

        action = dist.sample()[0]

        if type(self.action_dim) == list:
            action = action.detach().cpu().numpy() # Using only this is problematic for super mario since it returns a 0-size np array in discrete action space
            output_action = np.clip(action, self.action_dim[0], self.action_dim[1])
        else:
            output_action = action.item()

        return output_action

    def load_state(self, path):
        if os.path.exists(path):
            # if trained on gpu but test on cpu use:
            self.policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def save_state(self, path):
        torch.save(self.policy.state_dict(), path)


    def train_agent(self, env_builder, parallel=8, num_steps=128, epochs=3, batch_size=32):
        from train_scripts.EnvBuilder import MultiEnviroment
        multi_env = MultiEnviroment(env_builder, parallel)
        batch_states = np.zeros((parallel, num_steps + 1) + self.state_dim)
        batch_actions = np.zeros((parallel, num_steps) + self.action_dim)
        batch_values = np.zeros((parallel, num_steps, 1))
        batch_logprobs = np.zeros((parallel, num_steps, 1))
        batch_rewards = np.zeros((parallel, num_steps, 1))
        batch_dones = np.full((parallel, num_steps, 1), False)

        batch_states[:,0] = multi_env.reset()

        while True:
            for step in range(num_steps):
                dists, values = self.policy(torch.from_numpy(batch_states[:,step]).to(device).float())
                actions = dists.sample()
                batch_logprobs[:, step] = dists.log_prob(actions).detach().numpy()
                batch_actions[:, step] = actions.detach().numpy()
                batch_values[:, step] = values.detach().numpy()
                batch_states[:, step + 1], batch_rewards[:, step], batch_dones[:, step], _ = multi_env.step(batch_actions[:,step])


            _, next_values = self.policy(torch.from_numpy(batch_states[:, num_steps]).to(device).float()) # Todo: avoid runing twice

            batch_values_flatten = batch_values.reshape(-1, 1)
            batch_next_values_flatten  = np.concatenate((batch_values_flatten[1:], next_values), axis=0)
            batch_dones_flatten = batch_dones.reshape(-1, 1)
            batch_rewards_flatten = batch_rewards.reshape(-1, 1)
            batch_states_flatten = batch_states.reshape(-1, batch_states.shape[2])
            batch_actions_flatten = batch_actions.reshape(-1, batch_actions.shape[2])
            batch_log_probs_flatten = batch_logprobs.reshape(-1, 1)

            deltas = batch_rewards_flatten + self.hp['discount'] * batch_next_values_flatten * (1 - batch_dones_flatten) - batch_values_flatten
            advantages = monte_carlo_reward(deltas, batch_dones_flatten, self.hp['horizon'] * self.hp['discount'], device)
            rewards = advantages + batch_values_flatten
            advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-6)

            # Optimize policy for K epochs:
            dataset = NonSequentialDataset(batch_states_flatten, batch_values_flatten, batch_actions_flatten, batch_log_probs_flatten, rewards, advantages)
            dataloader = DataLoader(dataset, batch_size=self.hp['minibatch_size'], shuffle=True)
            for _ in range(self.hp['epochs']):
                for (states_batch, old_policy_values_batch, old_policy_actions_batch, old_policy_loggprobs_batch,
                     rewards_batch, advantages_batch) in dataloader:
                    # Evaluating old actions and values with the target policy:
                    dists, values = self.policy(states_batch)
                    values = values.view(-1)
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
                    logprobs = dists.log_prob(old_policy_actions_batch)
                    ratios = torch.exp(logprobs.view(-1) - old_policy_loggprobs_batch)
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

                    self.reporter.update_agent_stats("actor_loss", None, actor_loss.mean().item())
                    self.reporter.update_agent_stats("critic_loss", None, critic_loss.mean().item())
                    self.reporter.update_agent_stats("dist_entropy", None, -exploration_loss.mean().item())
                    self.reporter.update_agent_stats("ratios", None, ratios.mean().item())
                    self.reporter.update_agent_stats("values", None, values.mean().item())