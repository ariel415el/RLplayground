###############################################
# Credit to https://github.com/sfujim/TD3.git #
###############################################
import os
from Agents.dnn_models import *
import copy
from utils.utils import update_net
from utils.replay_memory import ListMemory
from Agents.GenericAgent import GenericAgent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device: ", device)


class TD3_paper_actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1):
        super(TD3_paper_actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class TD3_paper_critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TD3_paper_critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(GenericAgent):
    def __init__(self, state_dim, action_space, bounderies, hp, train=True):
        super(TD3, self).__init__(train)
        self.action_space = action_space
        self.state_dim = state_dim
        self.bounderies = torch.tensor(bounderies).float().to(device)
        self.action_dim = len(bounderies[0])
        self.train = train
        self.steps=0

        self.hp = {
            'tau':0.005,
            'actor_lr':0.0003,
            'critic_lr':0.0003,
            'lr_decay':0.999,
            'discount':0.99,
            'policy_update_freq':2,
            'batch_size':256,
            'max_playback':1000000,
            'exploration_steps':10000,
            'min_memory_for_learning':10000,
            'policy_noise_sigma':0.2,
            'noise_clip':0.5,
            'exploration_noise_sigma':0.1,
        }
        self.hp.update(hp)
        self.playback_memory = ListMemory(self.hp['max_playback'])

        self.trainable_actor = TD3_paper_actor(self.state_dim[0], self.action_dim).to(device)
        self.target_actor = copy.deepcopy(self.trainable_actor)
        self.actor_optimizer = torch.optim.Adam(self.trainable_actor.parameters(), lr=self.hp['actor_lr'])

        self.trainable_critics = TD3_paper_critic(self.state_dim[0], self.action_dim).to(device)
        self.target_critics = copy.deepcopy(self.trainable_critics)
        self.critics_optimizer = torch.optim.Adam(self.trainable_critics.parameters(), lr=self.hp['critic_lr'])

        self.name = "TD3_lr[%.4f]_b[%d]_tau[%.4f]_uf[%d]"%(
            self.hp['actor_lr'], self.hp['batch_size'],
            self.hp['tau'], self.hp['policy_update_freq'])

    def process_new_state(self, state):
        self.trainable_actor.eval()
        with torch.no_grad():
            state_torch = torch.from_numpy(state).to(device).float().view(1,-1)
            action = self.trainable_actor(state_torch).cpu().data.numpy()[0]
        self.trainable_actor.train()
        if self.train:
            if self.steps < self.hp['exploration_steps']:
                # action = np.random.uniform(-1, 1,size=action.shape) # TODO: use self.bounderies
                action = self.action_space.sample()
            else:
                action += np.random.normal(0, self.hp['exploration_noise_sigma'], size=action.shape)

        self.last_state = state
        self.last_action = action

        action = np.clip(action, self.bounderies[0].cpu().numpy(), self.bounderies[1].cpu().numpy())
        return action

    def process_output(self, new_state, reward, is_finale_state):
        self.steps += 1
        if self.train:
            self.playback_memory.add_sample((self.last_state, self.last_action, new_state, reward, is_finale_state))
            if self.steps > self.hp['exploration_steps']:
                self._learn()
                if self.steps % self.hp['policy_update_freq'] == 0:
                    update_net(self.target_actor, self.trainable_actor, self.hp['tau'])
                    update_net(self.target_critics, self.trainable_critics, self.hp['tau'])

                    # # Decay lr
                    # for param_group in self.actor_optimizer.param_groups:
                    #     param_group['lr'] *= self.hp['lr_decay']
                    # for param_group in self.critics_optimizer.param_groups:
                    #     param_group['lr'] *= self.hp['lr_decay']
                    # self.reporter.add_costume_log("actor_lr", self.steps, self.actor_optimizer.param_groups[0]['lr'])
                    # self.reporter.add_costume_log("critic_lr", self.steps, self.critics_optimizer.param_groups[0]['lr'])

    def _learn(self):
        if len(self.playback_memory) > self.hp['min_memory_for_learning']:
            states, actions, next_states, rewards, is_finale_states = self.playback_memory.sample(self.hp['batch_size'], device)
            # update critics
            states = states.float()
            next_states = next_states.float()
            with torch.no_grad():
                next_action = self.target_actor(next_states)
                # noise = torch.empty(next_action.shape).normal_(mean=0,std=self.hp['policy_noise_sigma']).clamp(-self.hp['noise_clip'], self.hp['noise_clip']).to(device)
                noise = (torch.randn_like(actions) * self.hp['policy_noise_sigma']).clamp(-self.hp['noise_clip'], self.hp['noise_clip']).to(device)
                next_noisy_action = next_action + noise
                next_noisy_action = torch.max(torch.min(next_noisy_action, self.bounderies[1]), self.bounderies[0])
                next_target_q_values_1, next_target_q_values_2 = self.target_critics(next_states, next_noisy_action)
                next_target_q_values = torch.min(next_target_q_values_1.view(-1) , next_target_q_values_2.view(-1))

                not_final = (1-is_finale_states.float())
                target_values = rewards + self.hp['discount']*next_target_q_values*not_final


            # update critics
            self.trainable_critics.train()
            self.critics_optimizer.zero_grad()
            q_values_1, q_values_2 = self.trainable_critics(states, actions)
            critic_loss = ( (q_values_1.view(-1) - target_values).pow(2) + (q_values_2.view(-1) - target_values).pow(2) ).mean()
            critic_loss.backward()
            self.critics_optimizer.step()
            self.reporter.add_costume_log("critic_loss", self.steps, critic_loss.item())

            # # update  policy only each few steps (delayed update)
            if self.steps % self.hp['policy_update_freq'] == 0:
                # update actor
                self.actor_optimizer.zero_grad()
                actor_obj = -self.trainable_critics.Q1(states, self.trainable_actor(states)).mean() # paper suggest using critic_1 ?
                actor_obj.backward()
                self.actor_optimizer.step()
                self.reporter.add_costume_log("actor_obj", self.steps, actor_obj.item())

    def load_state(self, path):
        if path is not None and os.path.exists(path):
            dict = torch.load(path, map_location=lambda storage, loc: storage)

            self.trainable_actor.load_state_dict(dict['actor'])
            self.trainable_critics.load_state_dict(dict['critic'])
        else:
            print("Couldn't find weights file")

    def save_state(self, path):
        dict = {'actor':self.trainable_actor.state_dict(), 'critic': self.trainable_critics.state_dict()}
        torch.save(dict, path)
