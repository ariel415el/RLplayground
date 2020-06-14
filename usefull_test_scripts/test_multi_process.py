from multiprocessing import Process, Pipe
import gym
import numpy as np
from time import time

class Environment(Process):
    def __init__(
            self,
            env_idx,
            child_conn):
        super(Environment, self).__init__()
        self.env = gym.make('LunarLander-v2')
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.child_conn = child_conn
        self.reset()

    def run(self):
        super(Environment, self).run()
        while True:
            action = self.child_conn.recv()

            obs, reward, done, info = self.env.step(action)

            self.rall += reward
            self.steps += 1

            if done:
                self.history = self.reset()

            self.child_conn.send(
                [obs, reward, done, info])

    def reset(self):
        self.steps = 0
        self.episode += 1
        self.rall = 0
        obs = self.env.reset()
        return obs

def dummy_compute():
    x = 0
    for i in range(100):
        for j in range(100):
            x = (x + 32)/ 4

def run_multi_process(num_workers):
    action_space = gym.make('LunarLander-v2').action_space
    # states = np.zeros([num_workers, 8])
    steps_in_batch = 4000
    workers = []
    parent_conns = []
    child_conns = []
    for idx in range(num_workers):
        parent_conn, child_conn = Pipe()
        worker = Environment(idx, child_conn)
        worker.start()
        workers.append(worker)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    for s in range(steps_in_batch):
        actions = [action_space.sample() for _ in range(num_workers)]
        dummy =  [dummy_compute() for _ in range(num_workers)]
        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        next_states, rewards, dones = [], [], []
        for parent_conn in parent_conns:
            s, r, d, _ = parent_conn.recv()
            # next_states.append(s)
            # rewards.append(r)
            # dones.append(d)
    for idx in range(num_workers):
        workers[idx].terminate()

def run_single_process(num_workers):
    env = gym.make('LunarLander-v2')
    env.reset()
    steps_in_batch = 4000
    for w in range(num_workers):
        for s in range(steps_in_batch):
            action = env.action_space.sample()
            dummy_compute()
            s, r, d, _ = env.step(action)
            if d:
                env.reset()

if __name__ == '__main__':
    s = time()
    run_multi_process(4)
    print("run_multi_process took %f seconds"%(time()-s))

    s = time()
    run_single_process(4)
    print("run_single_process took %f seconds"%(time()-s))