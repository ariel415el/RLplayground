import gym
import pybulletgym
from AtariEnvWrappers import get_final_env
from Agents.discrete_agents import *
from Agents.continous_agents import *
from Agents.hybrid_agents import *

env_goals = {"CartPole-v1":195, "Acrobot-v1":-80, "MountainCar-v0":-110, "Pendulum-v0":-200, "LunarLander-v2":200,
             "LunarLanderContinuous-v2": 200, "BipedalWalker-v3":500, "BipedalWalkerHardcore-v3":500,
             "PongNoFrameskip-v4":20, "BreakoutNoFrameskip-v4":200,
             'AntPyBulletEnv-v0':6000, "Walker2DMuJoCoEnv-v0":6000, 'HumanoidMuJoCoEnv-v0':6000, 'HalfCheetahMuJoCoEnv-v0':6000 }


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
    elif agent_name == "PPO_ICM":
        agent = PPO_ICM.HybridPPO_ICM(state_dim, action_dim, hp)
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

def get_regular_env_setting(env_name):
    env = gym.make(env_name)
    return env, env_goals[env_name]


def get_atari_env_setting(env_name, frame_stack):
    env = get_final_env(env_name, frame_stack=frame_stack)
    return env, env_goals[env_name]

def get_env_settings(env_name):
    if env_name == "PongNoFrameskip-v4":
        return get_atari_env_setting(env_name,frame_stack=1)
    elif env_name == "BreakoutNoFrameskip-v4":
        return get_atari_env_setting(env_name, frame_stack=2)
    else:
        return get_regular_env_setting(env_name)