import gym
import pybulletgym
from EnvWrappers import get_atari_env, get_super_mario_env, get_grid_maze_env
from Agents.discrete_agents import *
from Agents.continous_agents import *
from Agents.hybrid_agents import *

env_goals = {"CartPole-v1":195, "Acrobot-v1":-80, "MountainCar-v0":-110, "Pendulum-v0":-200, "LunarLander-v2":200,
             "LunarLanderContinuous-v2": 200, "BipedalWalker-v3":300, "BipedalWalkerHardcore-v3":300,
             "PongNoFrameskip-v4":20, "BreakoutNoFrameskip-v4":200,
             'AntPyBulletEnv-v0':6000, "Walker2DMuJoCoEnv-v0":6000, 'HumanoidMuJoCoEnv-v0':6000, 'HalfCheetahMuJoCoEnv-v0':6000,
             'SuperMarioBros-1':5000, 'SuperMarioBros-v2':5000, 'SuperMarioBros-v3':5000,
             "MiniGrid-FourRooms-v0":10}

class env_builder(object):
    def __init__(self,constructor, args):
        self.constructor = constructor
        self.args = args
    def __call__(self):
        return self.constructor(**self.args)

class MultiEnviroment(object):
    def __init__(self,env_builder, num_envs=1):
        self.num_envs = num_envs
        self.envs = [env_builder() for _ in range(num_envs)]

    def reset(self):
        return [env.reset() for env in self.envs]

    def step(self, actions):
        states = []
        rewards = []
        dones = []
        infos = []
        for i in range(self.num_envs):
            state, reward, done, info = self.envs[i].step(actions[i])
            states += [state]
            rewards += [reward]
            dones += [done]
            infos += [info]
        return np.array(states), np.array(rewards), np.array(dones), np.array(infos)



def build_agent(agent_name, env,  hp):
    state_dim, action_dim = get_state_and_action_dim(env)
    if agent_name == "DQN":
        agent = DQN_agent.DQN_agent(state_dim, action_dim, hp, double_dqn=True, dueling_dqn=False,
                                    prioritized_memory=False, noisy_MLP=False)
    elif agent_name == "VanilaPG":
        agent = VanilaPolicyGradient.VanilaPolicyGradient(state_dim, action_dim, hp)
    elif agent_name == "A2C":
        agent = GenericActorCritic.ActorCritic(state_dim, action_dim, hp)
    elif agent_name == "PPO":
        agent = PPO.HybridPPO(state_dim, action_dim, hp)
    elif agent_name == "DDPG":
        agent = DDPG.DDPG(state_dim, action_dim, hp)
    elif agent_name == "TD3":
        agent = TD3.TD3(state_dim, env.action_space, action_dim, hp, train=True)
    else:
        raise Exception("Agent: $s not supported for this ebviroment" % agent_name)

    return agent

def get_state_and_action_dim(env):
    state_dim = env.observation_space.shape
    if type(env.action_space) == gym.spaces.Discrete:
        action_dim = env.action_space.n
    else:
        action_dim = [env.action_space.low, env.action_space.high]
    return state_dim, action_dim


def get_env_builder(env_name):
    if env_name == "PongNoFrameskip-v4":
        return env_builder(get_atari_env, {'env_name':env_name, 'frame_stack':1})
    elif env_name == "BreakoutNoFrameskip-v4":
        return env_builder(get_atari_env, {'env_name':env_name,'frame_stack':4, 'episode_life':True, 'no_op_reset':False, 'disable_noop':True})
    elif "SuperMarioBros" in env_name:
        return env_builder(get_super_mario_env, {'env_name':env_name})
    elif "MiniGrid" in env_name:
        return env_builder(get_grid_maze_env, {'env_name':env_name})
    else:
        return env_builder(gym.make, {'id':env_name})

def get_env_goal(env_name):
    return env_goals[env_name]