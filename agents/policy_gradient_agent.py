import torch
import random
import numpy as np
import os
from _collections import deque
from agents.dnn_models import MLP_softmax

def get_action_vec(action, dim):
    res = np.zeros((dim, 1))
    res[action, 0] = 1
    return res

class policy_gradient_agent(object):
    def __init__(self, state_dim, action_dim, max_episodes,
                 train=True,
                 batch_size=64,
                 discount=0.99,
                 star_lr=0.001,
                 start_epsilon=1.0,
                 min_lr=0.0001,
                 min_epsilon=0.01,
                 epsilon_decay=0.998):
        self.name = "baseline-PG"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_episodes = max_episodes
        self.train=train
        self.batch_size = batch_size
        self.epsilon = start_epsilon
        self.discount = discount
        self.start_lr = star_lr
        self.start_epsilon = start_epsilon
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        self.gs_num = 0
        self.completed_episodes = 0
        self.last_rewards = deque(maxlen = 1000)

        self.batch_episodes = []
        self.currently_building_episode = []
        self.device = torch.device("cpu")
        self.trainable_model = MLP_softmax(self.state_dim, self.action_dim, [20, 10]).double()

        self.optimizer = torch.optim.Adam(self.trainable_model.parameters(), lr=star_lr)
        self.optimizer.zero_grad()


    def learn(self):
        self.optimizer.zero_grad()
        mean_reward = np.mean(self.last_rewards)
        for rollout in self.batch_episodes:
            b = 0
            Rt = 0
            for t in range(len(rollout) - 1, -1, -1):
                d = self.discount**(len(rollout) - t - 1)
                Rt = Rt*d + rollout[t][2]
                b = b*d + mean_reward
                log_prob = -(rollout[t][3]*(Rt - b))/len(self.batch_episodes)
                log_prob.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, state):
        action_logits = self.trainable_model(torch.tensor([state]))[0]
        best_action_index = np.random.choice(self.action_dim, p=action_logits.detach().numpy())
        return best_action_index, action_logits[best_action_index]

    def process_new_state(self, prev_state, prev_action, reward, cur_state, is_finale_state):
        action_index = None
        if self.train:
            if prev_state is not None:
                action_index, action_prob = self.get_action(cur_state)
                self.currently_building_episode += [(prev_state, prev_action, reward, torch.log(action_prob))]

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

        self.last_rewards.append(reward)
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
            self.trainable_model.load_state_dict(torch.load(path))

    def save_state(self, path):
        torch.save(self.trainable_model.state_dict(), path)

    def get_stats(self):
        return "Gs: %d; Epsilon: %.5f; LR: %.5f avg-reward %f"%(self.gs_num, self.epsilon, self.optimizer.param_groups[0]['lr'],self.running_reward_avg)


