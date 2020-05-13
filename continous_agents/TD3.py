import torch
import random
from collections import deque
import numpy as np
import os
from dnn_models import *
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device: ", device)

def update_net(model_to_change, reference_model, tau):
    for target_param, local_param in zip(model_to_change.parameters(), reference_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class TD3(object):
    def __init__(self, state_dim, bounderies, max_episodes, train = True, action_space = None):
        self.action_space = action_space
        self.state_dim = state_dim
        self.bounderies = torch.tensor(bounderies).float().to(device)
        self.action_dim = len(bounderies[0])
        self.max_episodes = max_episodes
        self.train = train
        self.tau=0.005
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.discount = 0.99
        self.policy_update_freq = 2
        self.batch_size = 256
        self.max_playback = 1000000
        self.exploration_steps = 25000
        self.exploration_steps = -1

        self.policy_noise_sigma = 0.2
        self.noise_clip = 0.5
        self.exploration_noise_sigma = 0.1

        self.completed_episodes = 0
        self.steps=0
        self.playback_deque = deque(maxlen=self.max_playback)

        self.trainable_actor = TD3_paper_actor(self.state_dim, self.action_dim).to(device)
        self.target_actor = TD3_paper_actor(self.state_dim, self.action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.trainable_actor.parameters(), lr=self.actor_lr)

        self.trainable_critic_1 = TD3_paper_critic(self.state_dim, self.action_dim).to(device)
        self.target_critic_1 = TD3_paper_critic(self.state_dim, self.action_dim).to(device)
        self.critic_optimizer_1 = torch.optim.Adam(self.trainable_critic_1.parameters(), lr=self.critic_lr)

        self.trainable_critic_2 = TD3_paper_critic(self.state_dim, self.action_dim).to(device)
        self.target_critic_2 = TD3_paper_critic(self.state_dim, self.action_dim).to(device)
        self.critic_optimizer_2 = torch.optim.Adam(self.trainable_critic_2.parameters(), lr=self.critic_lr)

        update_net(self.target_actor, self.trainable_actor, 1)
        update_net(self.target_critic_1, self.trainable_critic_2, 1)
        update_net(self.target_critic_2, self.trainable_critic_2, 1)


        self.name = "TD3_lr[%.4f]_b[%d]_tau[%.4f]_uf[%d]"%(self.actor_lr, self.batch_size, self.tau, self.policy_update_freq)

    def process_new_state(self, state):
        self.trainable_actor.eval()
        with torch.no_grad():
            state_torch = torch.from_numpy(state).to(device).float().view(1,-1)
            action = self.trainable_actor(state_torch).cpu().data.numpy()[0]
        self.trainable_actor.train()
        if self.train:
            if self.steps < self.exploration_steps:
                action = np.random.uniform(-1, 1,size=action.shape) # TODO: use self.bounderies
                action = self.action_space.sample()
            else:
                action += np.random.normal(0, self.exploration_noise_sigma, size=action.shape)

        self.last_state = state
        self.last_action = action

        # action = action.detach().cpu().numpy()
        action = np.clip(action, self.bounderies[0].cpu().numpy(), self.bounderies[1].cpu().numpy())
        return action

    def process_output(self, new_state, reward, is_finale_state):
        if self.train:
            self.playback_deque.append((self.last_state, self.last_action, new_state, reward, is_finale_state))
            if self.steps > self.exploration_steps:
                self._learn()
                if self.steps % self.policy_update_freq == 0:
                    update_net(self.target_actor, self.trainable_actor, self.tau)
                    update_net(self.target_critic_1, self.trainable_critic_1, self.tau)
                    update_net(self.target_critic_2, self.trainable_critic_2, self.tau)
        self.steps += 1

    def _learn(self):
        if len(self.playback_deque) > self.batch_size:
            batch_arrays = np.array(random.sample(self.playback_deque, k=self.batch_size))
            states = torch.from_numpy(np.stack(batch_arrays[:, 0], axis=0)).to(device).float()
            actions = torch.from_numpy(np.stack(batch_arrays[:, 1], axis=0)).to(device).float()
            next_states = torch.from_numpy(np.stack(batch_arrays[:, 2], axis=0)).to(device).float()
            rewards = torch.from_numpy(np.stack(batch_arrays[:, 3], axis=0)).to(device).float()
            is_finale_states = np.stack(batch_arrays[:, 4], axis=0)

            # update critics

            with torch.no_grad():
                next_action = self.target_actor(next_states)
                noise = torch.empty(next_action.shape).normal_(mean=0,std=self.policy_noise_sigma).clamp(-self.noise_clip, self.noise_clip).to(device)
                next_noisy_action = next_action + noise
                next_noisy_action = torch.max(torch.min(next_noisy_action, self.bounderies[0]), self.bounderies[1])
                next_target_q_values_1 = self.target_critic_1(next_states, next_noisy_action.float()).view(-1)
                next_target_q_values_2 = self.target_critic_2(next_states, next_noisy_action.float()).view(-1)
                next_target_q_values = torch.min(next_target_q_values_1, next_target_q_values_2)
                target_values = rewards
                mask = np.logical_not(is_finale_states)
                target_values[mask] += self.discount*next_target_q_values[mask]

            # update critic 1
            self.trainable_critic_1.train()
            self.critic_optimizer_1.zero_grad()
            q_values_1 = self.trainable_critic_1(states, actions)
            loss_1 = torch.nn.functional.mse_loss(q_values_1.view(-1), target_values)
            loss_1.backward()
            self.critic_optimizer_1.step()

            # update critic 2
            self.trainable_critic_2.train()
            self.critic_optimizer_2.zero_grad()
            q_values_2 = self.trainable_critic_2(states, actions)
            loss_2 = torch.nn.functional.mse_loss(q_values_2.view(-1), target_values)
            loss_2.backward()
            self.critic_optimizer_2.step()

            # update  policy only each few steps (delayed update)
            if self.steps % self.policy_update_freq == 0:
                # update actor
                self.actor_optimizer.zero_grad()
                actions = self.trainable_actor(states)
                actor_obj = -self.trainable_critic_1(states, actions).mean() # paper suggest using critic_1 ?
                actor_obj.backward()
                self.actor_optimizer.step()


    def load_state(self, path):
        if os.path.exists(path):
            dict = torch.load(path)
            self.trainable_actor.load_state_dict(dict['actor'])
            self.trainable_critic_1.load_state_dict(dict['critic_1'])
            self.trainable_critic_2.load_state_dict(dict['critic_2'])
        else:
            print("Couldn't find weights file")

    def save_state(self, path):
        dict = {'actor':self.trainable_actor.state_dict(), 'critic_1': self.trainable_critic_1.state_dict(), 'critic_2': self.trainable_critic_2.state_dict()}
        torch.save(dict, path)

    def get_stats(self):
        return "GS: %d; LR: a-%.5f\c-%.5f"%(self.steps, self.actor_optimizer.param_groups[0]['lr'],self.critic_optimizer_1.param_groups[0]['lr'])


