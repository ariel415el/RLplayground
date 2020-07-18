from Agents.discrete_agents import *
from Agents.continous_agents import *
from Agents.hybrid_agents import *
import gym

def build_agent(agent_name, env,  hp):
    state_dim, action_dim = get_state_and_action_dim(env) # TODO feed agents with action/space dim object
    if agent_name == "DQN":
        agent = DQN_agent.DQN_agent(state_dim, action_dim, hp, double_dqn=True, dueling_dqn=False,
                                    prioritized_memory=False, noisy_MLP=False)
    elif agent_name == "VanilaPG":
        agent = VanilaPolicyGradient.VanilaPolicyGradient(state_dim, action_dim, hp)
    elif agent_name == "A2C":
        agent = GenericActorCritic.ActorCritic(state_dim, action_dim, hp)
    elif agent_name == "PPO":
        agent = PPO.PPO(state_dim, action_dim, hp)
    elif agent_name == "PPOParallel":
        agent = PPO_parallel.PPOParallel(state_dim, action_dim, hp)
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