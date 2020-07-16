from torch.multiprocessing import Process, Pipe
import numpy as np

class EnvProcess(Process):
    """Runs a gym enviroment in a process forever while communicating inputs/outputs with parent process"""
    def __init__(self, env, child_conn, idx):
        super(EnvProcess, self).__init__()
        self.env = env
        self.child_conn = child_conn
        self.idx = idx
        self.initial_state = self.env.reset()

    def run(self):
        super(EnvProcess, self).run()
        while True:
            action = self.child_conn.recv()
            state, reward, done, info = self.env.step(action)
            if done: # Replace final state with start state
                state = self.env.reset()
            self.child_conn.send([state, reward, done, info])


    def get_initial_state(self):
        return self.initial_state

    def reset(self):
        raise NotImplementedError("No reset available")

class MultiEnviroment(object):
    """Manages multiple enviroment instances a-synchorniously"""
    def __init__(self, env_builder_instance, num_envs=1):
        self.env_builder = env_builder_instance
        self.num_envs = num_envs
        self.envs = []
        self.connections = []
        for i in range(num_envs):
            env = self.env_builder()
            parent_conn, child_conn = Pipe()
            self.connections += [parent_conn]
            proccess = EnvProcess(env, child_conn,i)
            proccess.start()
            self.envs += [proccess]

    def reset(self):
        raise NotImplementedError("No reset available")

    def get_initial_state(self):
        return np.array([np.array(env.initial_state) for env in self.envs])

    def step(self, actions):
        states = []
        rewards = []
        dones = []
        infos = []
        for i, conn in enumerate(self.connections):
            conn.send(actions[i])
        for i, conn in enumerate(self.connections):
            state, reward, done, info = conn.recv()
            states += [np.array(state)]
            rewards += [reward]
            dones += [done]
            infos += [info]

        return np.array(states), np.array(rewards), np.array(dones), np.array(infos)

    def close(self):
        for env in self.envs:
            env.close()
        self.envs = []


class MultiEnviromentSync(object):
    """Manages multiple enviroment instances """
    def __init__(self, env_builder_instance, num_envs=1):
        self.env_builder = env_builder_instance
        self.num_envs = num_envs
        self.envs = [self.env_builder() for _ in range(num_envs)]

    def reset(self):
        raise NotImplementedError("No reset available")

    def get_initial_state(self):
        return np.array([env.reset() for env in self.envs])

    def step(self, actions):
        states = []
        rewards = []
        dones = []
        infos = []
        for i in range(self.num_envs):
            state, reward, done, info = self.envs[i].step(actions[i])
            if done: # Replace final state with start state
                state = self.envs[i].reset()
            states += [state]
            rewards += [reward]
            dones += [done]
            infos += [info]

        return np.array(states), np.array(rewards), np.array(dones), np.array(infos)

    def close(self):
        for env in self.envs:
            env.close()
        self.envs = []
