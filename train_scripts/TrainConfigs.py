from train_scripts.EnvBuilder import get_env_settings, build_agent


def solve_cart_pole(agent_name):
    env_name = "CartPole-v1"
    env, solved_score = get_env_settings(env_name)
    agent_configs = {
        "DQN": {'lr':0.001, "min_playback":0, "max_playback":1000000, "update_freq": 100, 'hiden_layer_size':32, 'epsilon_decay':500},
        "VanilaPG": {'lr':0.001, 'batch_episodes':1},
        "A2C": {'lr':0.001, 'batch_episodes':1, 'GAE': 0.98},
        # "PPO": {'lr': 0.001, 'batch_episodes': 1, 'epochs': 4, 'GAE': 1.0, 'value_clip': 0.3, 'grad_clip': None},
        "PPO": {'lr': 0.001, 'batch_episodes': 10, 'epochs': 10,'minibatch_size':32, 'GAE': 1.0, 'epsilon_clio':0.1, 'value_clip': None, 'grad_clip': None},
        "PPO_ICM": {'lr': 0.0025, 'epsilon_clip': 0.3, 'batch_episodes': 4, 'epochs': 8, 'GAE': 0.95, 'value_clip': None,
              'grad_clip': None, 'use_extrinsic_reward': True, 'intrinsic_reward_scale': 1.0, 'lr_decay': 1.0}
    }
    agent = build_agent(agent_name, env, agent_configs[agent_name])
    return env_name, env, agent, solved_score


def solve_acrobot(agent_name):
    env_name = "Acrobot-v1"
    env, solved_score = get_env_settings(env_name)
    agent_configs = {
        "DQN": {'lr':0.001, "min_playback":0, "max_playback":1000000, "update_freq": 100, 'hiden_layer_size':32, 'epsilon_decay':500},
        "VanilaPG": {'lr':0.001, 'batch_episodes':1},
        "A2C": {'lr':0.001, 'batch_episodes':1, 'GAE': 0.98},
        "PPO": {'lr':0.001, 'epsilon_clip':0.3, 'batch_episodes':2, 'epochs':4, 'GAE':0.95, 'value_clip':None, 'grad_clip':None},
        "PPO_ICM": {'lr': 0.0025, 'epsilon_clip': 0.3, 'batch_episodes': 4, 'epochs': 8, 'GAE': 0.95, 'value_clip': None,
              'grad_clip': None, 'use_extrinsic_reward': True, 'intrinsic_reward_scale': 1.0, 'lr_decay': 1.0}
    }
    agent = build_agent(agent_name, env, agent_configs[agent_name])
    return env_name, env, agent, solved_score


def solve_mountain_car(agent_name):
    env_name = "MountainCar-v0";
    env, solved_score = get_env_settings(env_name)
    agent_configs = {
        "DQN": {'lr':0.0005, "min_playback":0, "max_playback":1000000, "update_freq": 100, 'hiden_layer_size':64,
                'epsilon_decay':500, 'batch_size':128, 'lr_decay':0.9999},
        "PPO_ICM": {'lr': 0.001, 'lr_decay': 0.9999, 'batch_episodes': 1, 'epochs': 3, 'GAE': 0.95, 'epsilon_clip': 0.2, 'value_clip': 0.2,
              'grad_clip': 0.5, 'hidden_layers': [64], 'intrinsic_reward_scale': 200.0, 'use_extrinsic_reward': False,
              'curiosity_hidden_dim': 32, 'entropy_weight': 0.01, 'curiosity_lr':0.00001}
    }
    agent = build_agent(agent_name, env, agent_configs[agent_name])
    return env_name, env, agent, solved_score


def solve_pendulum(agent_name):
    env_name = "Pendulum-v0"
    env, solved_score = get_env_settings(env_name)
    agent_configs = {
        "VanilaPG": {'lr': 0.0001, 'batch_episodes': 32, 'hidden_layers': [400, 300]},
        "A2C": {'lr':0.0004, 'lr_decay':0.99, 'batch_episodes':64, 'GAE':0.95, 'hidden_layers':[400,400]},
        "PPO": {'lr': 0.0004, "discount":0.99, 'lr_decay':0.9, 'batch_episodes': 10, 'epochs': 10, 'minibatch_size':1000, 'GAE': 0.95, 'epsilon_clip': 0.1, 'value_clip': 0.1,
                'grad_clip': None, 'entropy_weight': 0.01, 'hidden_layers':[400,400]},
        "PPO_ICM": {'lr': 0.001, 'batch_episodes': 8, 'epochs': 10, 'GAE': 0.95, 'epsilon_clip': 0.3, 'value_clip': 0.5,
              'grad_clip': 0.5, 'entropy_weight': 0.01, 'hidden_layers': [512]},
        "DDPG": {'actor_lr':0.0001, 'critic_lr':0.001, 'batch_size':64, 'min_playback':1000, 'layer_dims':[400,300],
                 'tau':0.001, "update_freq":1, 'learn_freq':1},
        "TD3": {'actor_lr':0.00005, 'critic_lr':0.0001, "exploration_steps":5000, "min_memory_for_learning":10000, "batch_size": 128}
    }
    agent = build_agent(agent_name, env, agent_configs[agent_name])

    return env_name, env, agent, solved_score


def solve_lunar_lander(agent_name):
    env_name = "LunarLander-v2"
    env, solved_score = get_env_settings(env_name)
    agent_configs = {
        "DQN": {'lr':0.0007, "min_playback":1000, "max_playback":1000000, "update_freq": 500, 'hiden_layer_size':256, 'epsilon_decay':10000},
        "VanilaPG": {'lr':0.001, 'batch_episodes':32, 'hidden_layers':[64,64,128]},
        "A2C": {'lr':0.005, 'batch_episodes':8, 'GAE': 0.96, 'hidden_layers':[64,32]},
        "PPO": {'lr':0.00025, 'batch_episodes':8, 'epochs':3, 'GAE':0.95, 'epsilon_clip': 0.1, 'value_clip':None,
              'grad_clip':0.5, 'entropy_weight':0.01, 'hidden_layers':[64,32]},
        "PPO_ICM": {'lr': 0.0005, 'lr_decay': 0.99, 'batch_episodes': 32, 'epochs': 10, 'GAE': 0.95, 'epsilon_clip': 0.25, 'value_clip': None, 'grad_clip': None,
              'entropy_weight': 0.01, 'hidden_dims': [400, 200, 200], 'curiosity_hidden_dim': 128}
    }
    agent = build_agent(agent_name, env, agent_configs[agent_name])
    return env_name, env, agent, solved_score

def solve_continous_lunar_lander(agent_name):
    env_name = "LunarLanderContinuous-v2";
    env, solved_score = get_env_settings(env_name)
    agent_configs = {
        "VanilaPG": {'lr': 0.001, 'batch_episodes': 32, 'hidden_layers': [64, 64, 128]},
        "A2C": {'lr': 0.005, 'lr_decay': 0.99, 'batch_episodes': 8, 'GAE': 0.96, 'hidden_layers': [400,200]},
        "PPO": {'lr': 0.00025, 'batch_episodes': 8, 'epochs': 3, 'GAE': 0.95, 'epsilon_clip': 0.1, 'value_clip': None,
                'grad_clip': 0.5, 'entropy_weight': 0.01, 'hidden_layers': [400,200]},
        "PPO_ICM": {'lr': 0.0005, 'lr_decay': 0.99, 'batch_episodes': 32, 'epochs': 10, 'GAE': 0.95,'epsilon_clip': 0.25,
                    'value_clip': None, 'grad_clip': None, 'entropy_weight': 0.01, 'hidden_dims': [400, 200, 200], 'curiosity_hidden_dim': 128},
        "DDPG": {'actor_lr': 0.0001, 'critic_lr': 0.001, 'batch_size': 100, 'min_playback': 0,
                 'layer_dims': [400, 200],'tau': 0.001, "update_freq": 1, 'learn_freq': 1},
        "TD3": {'actor_lr': 0.0003, 'critic_lr': 0.00025, "exploration_steps": 5000,  "min_memory_for_learning": 10000, "batch_size": 128}
    }
    agent = build_agent(agent_name, env, agent_configs[agent_name])

    return env_name, env, agent, solved_score


def solve_bipedal_walker(agent_name):
    env_name = "BipedalWalker-v3"
    env, solved_score = get_env_settings(env_name)
    agent_configs = {
        "VanilaPG":{'lr': 0.001, 'batch_episodes': 32, 'hidden_layers': [64, 64, 128]},
        "A2C":  {'lr': 0.005, 'batch_episodes': 8, 'GAE': 0.96, 'hidden_layers':[32,16]},
        "PPO": {'lr': 0.001, 'batch_episodes': 45, 'epochs': 10, 'GAE': 0.95, 'epsilon_clip': 0.1, 'value_clip': 0.1,
              'grad_clip': None, 'entropy_weight': 0.01, 'hidden_layers': 512},
        "PPO_ICM": {'lr': 0.0005, 'lr_decay': 0.99, 'batch_episodes': 32, 'epochs': 10, 'GAE': 0.95, 'epsilon_clip': 0.25,
              'value_clip': None, 'grad_clip': None, 'entropy_weight': 0.01, 'hidden_dims': [400, 200, 200], 'curiosity_hidden_dim': 128},
        "DDPG":{'actor_lr':0.0001, 'critic_lr':0.001, 'batch_size':100, 'min_playback':0, 'layer_dims':[400,200], 'tau':0.001, "update_freq":1, 'learn_freq':1},
        "TD3":  {'actor_lr':0.00025, 'critic_lr':0.00025}#, "exploration_steps":5000, "min_memory_for_learning":10000, "batch_size": 256}
    }
    agent = build_agent(agent_name, env, agent_configs[agent_name])

    return env_name, env, agent, solved_score



def solve_pong(agent_name):
    env_name = "PongNoFrameskip-v4"
    env, solved_score = get_env_settings(env_name)
    agent_configs = {
        "DQN":{'lr':0.0001, "min_playback":1000, "max_playback":100000, "update_freq": 1000, 'hiden_layer_size':512, "normalize_state":True, 'epsilon_decay':30000},
        "PPO": {'lr': 0.0001, 'batch_episodes': 8, 'epochs': 4, 'GAE': 1.0, 'epsilon_clip': 0.2, 'value_clip': None,
              'grad_clip': None, 'entropy_weight': 0.01, 'hidden_dims': [400, 200]},
    }
    agent = build_agent(agent_name, env, agent_configs[agent_name])

    return env_name, env, agent, solved_score



def solve_breakout(agent_name):
    env_name = "BreakoutNoFrameskip-v4"
    env, solved_score = get_env_settings(env_name)
    agent_configs = {
        "DQN":{'lr': 0.00001, "min_playback": 50000, "max_playback": 1000000, "update_freq": 10000, 'learn_freq': 4,
              "normalize_state": True, 'epsilon_decay': 5000000},
        "PPO": {'lr': 0.00005, 'batch_episodes': 16, 'epochs': 8, 'GAE': 0.98, 'epsilon_clip': 0.1, 'value_clip': 0.1,
              'grad_clip': None, 'entropy_weight': 0.01, 'hidden_dims': [400, 200]},
    }
    agent = build_agent(agent_name, env, agent_configs[agent_name])


def solve_half_cheetah(agent_name):
    env_name = 'HalfCheetahMuJoCoEnv-v0'
    env, solved_score = get_env_settings(env_name)
    agent_configs = {
        "VanilaPG": {'lr': 0.0001, 'batch_episodes': 32, 'hidden_layers': [400, 300]},
        "A2C": {'lr':0.0001, 'batch_episodes':64, 'GAE':0.95, 'hidden_layers':[400,400]},
        "PPO": {'lr': 0.001, 'batch_episodes': 45, 'epochs': 10, 'GAE': 0.95, 'epsilon_clip': 0.1, 'value_clip': 0.1,
              'grad_clip': None, 'entropy_weight': 0.01, 'hidden_layer_size': 512},
        "PPO_ICM": {'lr': 0.001, 'batch_episodes': 45, 'epochs': 10, 'GAE': 0.95, 'epsilon_clip': 0.1, 'value_clip': 0.1,
              'grad_clip': None, 'entropy_weight': 0.01, 'hidden_layer_size': 512},
        "DDPG": {'actor_lr':0.0001, 'critic_lr':0.001, 'batch_size':64, 'min_playback':1000, 'layer_dims':[400,300],
                 'tau':0.001, "update_freq":1, 'learn_freq':1},
        "TD3": {'actor_lr':0.0005, 'critic_lr':0.0005, "exploration_steps":1000, "min_memory_for_learning":5000, "batch_size": 64}
    }
    agent = build_agent(agent_name, env, agent_configs[agent_name])

    return env_name, env, agent, solved_score

def solve_humanoid(agent_name):
    env_name = 'HalfCHumanoidMuJoCoEnv-v0'
    env, solved_score = get_env_settings(env_name)
    agent_configs = {
        "VanilaPG": {'lr': 0.0001, 'batch_episodes': 32, 'hidden_layers': [400, 300]},
        "A2C": {'lr':0.0001, 'batch_episodes':64, 'GAE':0.95, 'hidden_layers':[400,400]},
        "PPO": {'lr': 0.001, 'batch_episodes': 45, 'epochs': 10, 'GAE': 0.95, 'epsilon_clip': 0.1, 'value_clip': 0.1,
              'grad_clip': None, 'entropy_weight': 0.01, 'hidden_layer_size': 512},
        "PPO_ICM": {'lr': 0.001, 'batch_episodes': 45, 'epochs': 10, 'GAE': 0.95, 'epsilon_clip': 0.1, 'value_clip': 0.1,
              'grad_clip': None, 'entropy_weight': 0.01, 'hidden_layer_size': 512},
        "DDPG": {'actor_lr':0.0001, 'critic_lr':0.001, 'batch_size':64, 'min_playback':1000, 'layer_dims':[400,300],
                 'tau':0.001, "update_freq":1, 'learn_freq':1},
        "TD3": {'actor_lr':0.0005, 'critic_lr':0.0005, "exploration_steps":1000, "min_memory_for_learning":5000, "batch_size": 64}
    }
    agent = build_agent(agent_name, env, agent_configs[agent_name])

    return env_name, env, agent, solved_score

def solve_2d_walker(agent_name):
    env_name = 'Walker2DMuJoCoEnv-v0'
    env, solved_score = get_env_settings(env_name)
    agent_configs = {
        "VanilaPG": {'lr': 0.0001, 'batch_episodes': 32, 'hidden_layers': [32, 32]},
        "A2C": {'lr':0.0001, 'batch_episodes':16, 'GAE':0.95, 'hidden_layers':[400,200]},
        "PPO": {'lr': 0.001,'lr_decay':0.9, 'batch_episodes': 45, 'epochs': 10, 'GAE': 0.95, 'epsilon_clip': 0.1, 'value_clip': 0.1,
              'grad_clip': None, 'entropy_weight': 0.01, 'hidden_layer_size': 512},
        "PPO_ICM": {'lr': 0.001, 'batch_episodes': 45, 'epochs': 10, 'GAE': 0.95, 'epsilon_clip': 0.1, 'value_clip': 0.1,
              'grad_clip': None, 'entropy_weight': 0.01, 'hidden_layer_size': 512},
        "DDPG": {'actor_lr':0.0001, 'critic_lr':0.001, 'batch_size':64, 'min_playback':1000, 'layer_dims':[400,300],
                 'tau':0.001, "update_freq":1, 'learn_freq':1},
        "TD3": {'actor_lr':0.0005, 'critic_lr':0.0005, "exploration_steps":1000, "min_memory_for_learning":5000, "batch_size": 64}
    }
    agent = build_agent(agent_name, env, agent_configs[agent_name])

    return env_name, env, agent, solved_score

def solve_ant(agent_name):
    env_name = 'AntPyBulletEnv-v0'
    env, solved_score = get_env_settings(env_name)
    agent_configs = {
        "VanilaPG": {'lr': 0.0001, 'batch_episodes': 32, 'hidden_layers': [400, 300]},
        "A2C": {'lr':0.0001, 'batch_episodes':64, 'GAE':0.95, 'hidden_layers':[400,400]},
        "PPO": {'lr': 0.001, 'batch_episodes': 45, 'epochs': 10, 'GAE': 0.95, 'epsilon_clip': 0.1, 'value_clip': 0.1,
              'grad_clip': None, 'entropy_weight': 0.01, 'hidden_layer_size': 512},
        "PPO_ICM": {'lr': 0.001, 'batch_episodes': 45, 'epochs': 10, 'GAE': 0.95, 'epsilon_clip': 0.1, 'value_clip': 0.1,
              'grad_clip': None, 'entropy_weight': 0.01, 'hidden_layer_size': 512},
        "DDPG": {'actor_lr':0.0001, 'critic_lr':0.001, 'batch_size':64, 'min_playback':1000, 'layer_dims':[400,300],
                 'tau':0.001, "update_freq":1, 'learn_freq':1},
        "TD3": {'actor_lr':0.0005, 'critic_lr':0.0005, "exploration_steps":1000, "min_memory_for_learning":5000, "batch_size": 64}
    }
    agent = build_agent(agent_name, env, agent_configs[agent_name])

    return env_name, env, agent, solved_score

def solve_super_mario(agent_name):
    env_name = 'SuperMarioBros-v3'
    env, solved_score = get_env_settings(env_name)
    agent_configs = {
        "DQN":{'lr':0.0001,'batch_size':1,'learn_freq':9999999, "min_playback":1000, "max_playback":100000, "update_freq": 1000, 'hiden_layer_size':16, "normalize_state":True, 'epsilon_decay':30000},
        "PPO": {'lr': 0.0001, 'batch_episodes': 8, 'epochs': 4, 'GAE': 1.0, 'epsilon_clip': 0.2, 'value_clip': None,
              'grad_clip': None, 'entropy_weight': 0.01, 'hidden_dims': [400, 200]},
    }
    agent = build_agent(agent_name, env, agent_configs[agent_name])
    return env_name, env, agent, solved_score

def solve_grid_maze(agent_name):
    env_name = "MiniGrid-FourRooms-v0"
    env, solved_score = get_env_settings(env_name)
    agent_configs = {

    }
    agent = build_agent(agent_name, env, agent_configs[agent_name])
    return env_name, env, agent, solved_score