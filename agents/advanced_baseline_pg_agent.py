import torch
import random
import numpy as np
import os
from agents.dnn_models import *

def get_action_vec(action, dim):
    res = np.zeros((dim, 1))
    res[action, 0] = 1
    return res

class actor_critic_agent(object):
    def __init__(self, state_dim, action_dim, max_episodes,
                 train=True,
                 batch_size=8,
                 discount=0.9999,
                 star_lr=0.01,
                 start_epsilon=1.0,
                 min_lr=0.0001,
                 min_epsilon=0.01,
                 epsilon_decay=0.998):
        self.name = 'actor-critic'
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_episodes = max_episodes
        self.train= train
        self.batch_size = batch_size
        self.discount = discount
        self.start_lr = star_lr
        self.start_epsilon = start_epsilon
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        self.epsilon = start_epsilon
        self.gs_num = 0
        self.num_steps = 0
        self.running_reward_avg = 0

        self.batch_episodes = []
        self.currently_building_episode = []
        self.device = torch.device("cpu")
        self.actor = MLP_softmax(self.state_dim, self.action_dim, [256]).double()
        self.critic = MLP(self.state_dim, 1, [256]).double()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=star_lr)
        self.actor_optimizer.zero_grad()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=star_lr)
        self.critic_optimizer.zero_grad()


    def learn(self):
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        for rollout in self.batch_episodes:
            R = 0
            for t in range(len(rollout) - 1, -1, -1):
                s, _, r, log_prob, ns = rollout[t]
                R += r
                v_prev = self.critic(torch.tensor(s))[0]

                actor_obj = -log_prob*(R - v_prev) / len(self.batch_episodes)

                critic_obj = 0.5*(r + self.discount*self.critic(torch.tensor(ns))[0] - v_prev).pow(2)

                total_obj = actor_obj + critic_obj
                total_obj.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

    def get_action(self, state):
        action_logits = self.actor(torch.tensor([state]))[0]
        best_action_index = np.random.choice(self.action_dim, p=action_logits.detach().numpy())
        return best_action_index, action_logits[best_action_index]

    def process_new_state(self, prev_state, prev_action, reward, cur_state, is_finale_state):
        action_index = None
        if self.train:
            if prev_state is not None:
                action_index, action_prob = self.get_action(cur_state)
                self.currently_building_episode += [(prev_state, prev_action, reward, torch.log(action_prob), cur_state)]

            if is_finale_state:
                self.batch_episodes += [self.currently_building_episode]
                self.currently_building_episode = []

                # decay_lr
                self.completed_episodes += 1
                new_lr = self.start_lr + (1 - self.completed_episodes / self.max_episodes) *(self.start_lr - self.min_lr)
                for g in self.optimizer.param_groups:
                    g['lr'] = new_lr

                # decay epsoilon
                # self.epsilon = self.start_epsilon + (1 - self.completed_episodes / self.max_episodes) *(self.start_epsilon - self.min_epsilon)
                self.epsilon = max(self.min_epsilon, self.epsilon*self.epsilon_decay)

            if len(self.batch_episodes) == self.batch_size:
                self.learn()
                self.gs_num += 1
                self.batch_episodes = []

                if (self.gs_num + 1) % 5 == 0:
                    for g in self.actor_optimizer.param_groups:
                        g['lr'] *= 0.993
                    for g in self.critic_optimizer.param_groups:
                            g['lr'] *= 0.993

                if (self.gs_num + 1) % 5 == 0:
                    self.epsilon *= 0.993

        self.running_reward_avg = self.running_reward_avg*self.num_steps + reward
        self.num_steps += 1
        self.running_reward_avg /= self.num_steps
        if is_finale_state:
            return 0
        else:
            if action_index is not None :
                return action_index
            elif random.uniform(0, 1) < self.epsilon:
                return random.randint(0, self.action_dim - 1)
            else:
                action_index, _ =  self.get_action(cur_state)
                return action_index



    def load_state(self, path):
        if os.path.exists(path):
            dicts = torch.load(path)
            self.actor.load_state_dict(dicts['actor'])
            self.critic.load_state_dict(dicts['critic'])

    def save_state(self, path):
        torch.save({"actor":self.actor.state_dict(), "critic":self.critic_optimizer.state_dict()}, path)

    def get_stats(self):
        return "Gs: %d; Epsilon: %.5f; LR: %.5f\%.5f avg-reward %f"%(self.gs_num, self.epsilon, self.actor_optimizer.param_groups[0]['lr'], self.critic_optimizer.param_groups[0]['lr'],self.running_reward_avg)



