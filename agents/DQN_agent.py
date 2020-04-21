import torch
import random
from collections import deque
import numpy as np
import os
from agents.dnn_models import MLP

def get_action_vec(action, dim):
    res = np.zeros((dim, 1))
    res[action, 0] = 1
    return res


class DQN_agent(object):
    def __init__(self, state_dim, action_dim, max_episodes,
                train = True,
                batch_size = 64,
                discount = 0.99,
                learning_freq=1,
                update_freq=1,
                max_playback=50000,
                tau=0.003,
                star_lr=0.0005,
                start_epsilon=1.0,
                min_lr=0.0005,
                min_epsilon=0.01,
                epsilon_decay=0.995):

        self.name = "DQN"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_episodes = max_episodes
        self.tau=tau
        self.start_lr = star_lr
        self.start_epsilon = start_epsilon
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.train = train
        self.discount = discount
        self.learning_freq = learning_freq
        self.update_freq = update_freq
        self.batch_size = batch_size

        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.action_counter = 0
        self.completed_episodes = 0
        self.gs_num=0
        self.playback_deque = deque(maxlen=max_playback)

        self.device = torch.device("cpu")
        self.trainable_model = MLP(self.state_dim, self.action_dim, [64, 64]).double()
        with torch.no_grad():
            self.periodic_model = MLP(self.state_dim, self.action_dim, [64, 64]).double()

        self.optimizer = torch.optim.Adam(self.trainable_model.parameters(), lr=star_lr, weight_decay=1e-5)
        self.optimizer.zero_grad()

    def learn(self):
        # batch_size = min(self.batch_size, len(self.playback_deque))
        if len(self.playback_deque) > self.batch_size:
            batch_indices = random.sample(range(len(self.playback_deque)), self.batch_size)
            batch_arrays = np.array(self.playback_deque)[batch_indices]
            prev_states = np.stack(batch_arrays[:, 0], axis=0)
            prev_actions = np.stack(batch_arrays[:, 1], axis=0)
            rewards = np.stack(batch_arrays[:, 2], axis=0)
            next_states = np.stack(batch_arrays[:, 3], axis=0)
            is_finale_states = np.stack(batch_arrays[:, 4], axis=0)

            with torch.no_grad():
                net_outs = self.periodic_model(torch.tensor(next_states).double())
            target_values = torch.tensor(rewards)
            target_values[np.logical_not(is_finale_states)] += self.discount*net_outs.max(axis=1)[0][np.logical_not(is_finale_states)]
            # target_values = torch.tensor(rewards) + net_outs.max(axis=1)[0] * (torch.tensor(1 - is_finale_states))

            self.trainable_model.train()
            curr_q_vals = torch.matmul(self.trainable_model(torch.tensor(prev_states)).view(-1,1, self.action_dim).float(), torch.tensor(prev_actions).float()).view(-1,1)
            # curr_q_vals = self.trainable_model.forward(torch.tensor(prev_states)).gather(1, torch.tensor(prev_actions))

            loss = torch.nn.functional.mse_loss(curr_q_vals.double(), target_values.view(-1,1).double())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def get_action(self, state):
        if random.uniform(0,1) < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            self.trainable_model.eval()
            with torch.no_grad():
                q_vals = self.trainable_model(torch.tensor(state))
            return np.argmax(q_vals.detach().cpu().numpy())

    def process_new_state(self, prev_state, prev_action, reward, cur_state, is_finale_state):
        if self.train:
            if is_finale_state:
                self.completed_episodes += 1
                # new_lr = self.start_lr + (1 - self.completed_episodes / self.max_episodes) *(self.start_lr - self.min_lr)
                # for g in self.optimizer.param_groups:
                #     g['lr'] = new_lr

                # self.epsilon = self.min_epsilon + (1 - self.completed_episodes / self.max_episodes) *(self.start_epsilon - self.min_epsilon)
                self.epsilon = max(self.min_epsilon, self.epsilon*self.epsilon_decay)

            if prev_state is not None:
                self.playback_deque.append([prev_state, get_action_vec(prev_action, self.action_dim), reward, cur_state, is_finale_state])
                # self.playback_deque.append([prev_state, prev_action, reward, cur_state, is_finale_state])

            if self.action_counter % self.learning_freq == 0:
                self.learn()
                self.gs_num += 1
            if self.action_counter % self.update_freq == 0:
                for target_param, local_param in zip(self.periodic_model.parameters(), self.trainable_model.parameters()):
                    target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

                # periodic_state_dict = self.periodic_model.state_dict()
                # trainable_state_dict = self.trainable_model.state_dict()
                # new_state_dict = {s: self.tau*trainable_state_dict[s] + (1.0-self.tau)*periodic_state_dict[s] for s in periodic_state_dict}
                # self.periodic_model.load_state_dict(new_state_dict)
                # # self.periodic_model.load_state_dict(self.trainable_model.state_dict())

        self.action_counter += 1
        if is_finale_state:
            return 0
        else:
            return self.get_action(cur_state)


    def load_state(self, path):
        if os.path.exists(path):
            self.periodic_model.load_state_dict(torch.load(path))
            self.trainable_model.load_state_dict(torch.load(path))
        else:
            print("Couldn't find weights file")

    def save_state(self, path):
        torch.save(self.periodic_model.state_dict(), path)

    def get_stats(self):
        return "GS: %d, Epsilon: %.5f; LR: %.5f"%(self.gs_num, self.epsilon, self.optimizer.param_groups[0]['lr'])


