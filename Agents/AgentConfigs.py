"""
This file contain functions that returns enviroment factories and agent with according configurations
"""
env_goals = {"CartPole-v1":195, "Acrobot-v1":-80, "MountainCar-v0":-110, "Pendulum-v0":-200, "LunarLander-v2":200,
             "LunarLanderContinuous-v2": 200, "BipedalWalker-v3":300, "BipedalWalkerHardcore-v3":300,
             "PongNoFrameskip-v4":20, "BreakoutNoFrameskip-v4":200, 'BreakoutDeterministic-v4':200,
             'AntPyBulletEnv-v0':6000, "Walker2DMuJoCoEnv-v0":6000, 'HumanoidMuJoCoEnv-v0':6000, 'HalfCheetahMuJoCoEnv-v0':6000,
             'SuperMarioBros-1':5000, 'SuperMarioBros-v2':5000, 'SuperMarioBros-v3':5000,
             "MiniGrid-FourRooms-v0":10}

def get_agent_configs(agent_name, env_name):
    if env_name == "CartPole-v1":
        agent_configs = {
            "DQN": {'lr': 0.0007, "min_playback": 1000, "max_playback": 1000000, "update_freq": 500,
                    'hiden_layer_size': 256, 'epsilon_decay': 10000},
            "VanilaPG": {'lr': 0.001, 'batch_episodes': 32, 'hidden_layers': [64, 64, 128]},
            "A2C": {'lr': 0.005, 'batch_episodes': 8, 'GAE': 0.96, 'hidden_layers': [64, 32]},
            "PPO": {'lr': 0.001, 'batch_episodes': 8, 'epochs': 3, 'GAE': 0.95, 'minibatch_size': 32,
                    'epsilon_clip': 0.1, 'value_clip': None,
                    'grad_clip': 0.5, 'entropy_weight': 0.01, 'fe_layers': [64], 'model_layers': [32, 32]},
            "PPOParallel": {'lr': 0.001, 'lr_decay': 0.9999, 'concurrent_epsiodes': 8, 'horizon': 4000, 'epochs': 3,
                            'minibatch_size': 32,
                            'GAE': 0.95, 'epsilon_clip': 0.1, 'value_clip': None, 'grad_clip': 0.5, 'fe_layers': [64],
                            'model_layers': [32, 32]},
        }
    elif env_name == 'Acrobot-v1':
        agent_configs = {
            "DQN": {'lr': 0.001, "min_playback": 0, "max_playback": 1000000, "update_freq": 100, 'hiden_layer_size': 32,
                    'epsilon_decay': 500},
            "VanilaPG": {'lr': 0.001, 'batch_episodes': 1},
            "A2C": {'lr': 0.001, 'batch_episodes': 1, 'GAE': 0.98},
            "PPO": {'lr': 0.001, 'epsilon_clip': 0.3, 'batch_episodes': 2, 'epochs': 4, 'GAE': 0.95, 'value_clip': None,
                    'grad_clip': None},
            "PPO_ICM": {'lr': 0.0025, 'epsilon_clip': 0.3, 'batch_episodes': 4, 'epochs': 8, 'GAE': 0.95,
                        'value_clip': None,
                        'grad_clip': None, 'use_extrinsic_reward': True, 'intrinsic_reward_scale': 1.0, 'lr_decay': 1.0}
        }
    elif env_name == "MountainCar-v0":
        agent_configs = {
            "DQN": {'lr': 0.0005, "min_playback": 0, "max_playback": 1000000, "update_freq": 100,
                    'hiden_layer_size': 64,
                    'epsilon_decay': 500, 'batch_size': 128, 'lr_decay': 0.9999},
            "PPO": {'lr': 0.0001, "discount": 0.99, 'lr_decay': 0.999, 'batch_episodes': 1, 'epochs': 3,
                    'hidden_layers': [64, 64],
                    'minibatch_size': 128, 'GAE': 0.95, 'epsilon_clip': 0.1, 'value_clip': None, 'grad_clip': 0.5,
                    'entropy_weight': 0.01
                    # 'curiosity_hp': {'intrinsic_reward_scale': 200.0, 'hidden_dim': 128, 'lr': 0.001}

                    },
            "PPOParallel": {'lr': 0.002, 'lr_decay': 0.9999, 'concurrent_epsiodes': 16, 'horizon': 200, 'epochs': 3,
                            'minibatch_size': 100,
                            'GAE': 0.95, 'epsilon_clip': 0.2, 'value_clip': 0.2, 'grad_clip': 0.5,
                            'fe_layers': [64, 64], 'model_layers': [64]},
        }
    elif env_name == "Pendulum-v0":
        agent_configs = {
            "VanilaPG": {'lr': 0.0001, 'batch_episodes': 32, 'hidden_layers': [400, 300]},
            "A2C": {'lr': 0.0004, 'lr_decay': 0.99, 'batch_episodes': 64, 'GAE': 0.95, 'hidden_layers': [400, 400]},
            "PPO_ICM": {'lr': 0.001, 'batch_episodes': 8, 'epochs': 10, 'GAE': 0.95, 'epsilon_clip': 0.3,
                        'value_clip': 0.5,
                        'grad_clip': 0.5, 'entropy_weight': 0.01, 'hidden_layers': [512]},
            "DDPG": {'actor_lr': 0.0001, 'critic_lr': 0.001, 'batch_size': 64, 'min_playback': 1000,
                     'layer_dims': [400, 300],
                     'tau': 0.001, "update_freq": 1, 'learn_freq': 1},
            "TD3": {'actor_lr': 0.00005, 'critic_lr': 0.0001, "exploration_steps": 5000,
                    "min_memory_for_learning": 10000, "batch_size": 128}
        }

    elif env_name == "LunarLander-v2":
        agent_configs = {
            "DQN": {'lr': 0.0007, "min_playback": 1000, "max_playback": 1000000, "update_freq": 500,
                    'hiden_layer_size': 256, 'epsilon_decay': 10000},
            "VanilaPG": {'lr': 0.001, 'batch_episodes': 32, 'hidden_layers': [64, 64, 128]},
            "A2C": {'lr': 0.005, 'batch_episodes': 8, 'GAE': 0.96, 'hidden_layers': [64, 32]},
            "PPO": {'lr': 0.001, 'batch_episodes': 8, 'epochs': 3, 'GAE': 0.95, 'minibatch_size': 32,
                    'epsilon_clip': 0.1, 'value_clip': None,
                    'grad_clip': 0.5, 'entropy_weight': 0.01, 'fe_layers': [64], 'model_layers': [32, 32]},
            "PPOParallel": {'lr': 0.001, 'lr_decay': 0.9999, 'concurrent_epsiodes': 8, 'horizon': 4000, 'epochs': 3,
                            'minibatch_size': 32,
                            'GAE': 0.95, 'epsilon_clip': 0.1, 'value_clip': None, 'grad_clip': 0.5, 'fe_layers': [64],
                            'model_layers': [32, 32]},
        }

    elif env_name == "LunarLanderContinuous-v2":
        agent_configs = {
            "VanilaPG": {'lr': 0.001, 'batch_episodes': 32, 'hidden_layers': [64, 64, 128]},
            "A2C": {'lr': 0.005, 'lr_decay': 0.99, 'batch_episodes': 8, 'GAE': 0.96, 'hidden_layers': [400, 200]},
            "PPO": {'lr': 0.001, 'batch_episodes': 8, 'epochs': 3, 'GAE': 0.95, 'minibatch_size': 32,
                    'epsilon_clip': 0.1, 'value_clip': None,
                    'grad_clip': 0.5, 'entropy_weight': 0.01, 'fe_layers': [64], 'model_layers': [32, 32]},
            "PPOParallel": {'lr': 0.00025, 'concurrent_epsiodes': 16, 'horizon': 128, 'epochs': 3, 'minibatch_size': 32,
                            'GAE': 0.95, 'epsilon_clip': 0.1, 'value_clip': None, 'grad_clip': 0.5},
            "DDPG": {'actor_lr': 0.0001, 'critic_lr': 0.001, 'batch_size': 100, 'min_playback': 0,
                     'layer_dims': [400, 200], 'tau': 0.001, "update_freq": 1, 'learn_freq': 1},
            "TD3": {'actor_lr': 0.0003, 'critic_lr': 0.00025, "exploration_steps": 5000,
                    "min_memory_for_learning": 10000, "batch_size": 128}
        }



    elif env_name == "BipedalWalker-v3":
        agent_configs = {
            "VanilaPG": {'lr': 0.001, 'batch_episodes': 32, 'hidden_layers': [64, 64, 128]},
            "A2C": {'lr': 0.005, 'batch_episodes': 8, 'GAE': 0.96, 'hidden_layers': [32, 16]},
            "PPO": {'lr': 0.0001, 'lr_decay': 0.999, 'batch_episodes': 8, 'epochs': 16, 'minibatch_size': 512,
                    'GAE': 0.95, 'epsilon_clip': 0.2, 'value_clip': None,
                    'grad_clip': 0.5, 'entropy_weight': 0.01, 'hidden_layers': [256, 128]},
            "PPO_ICM": {'lr': 0.0005, 'lr_decay': 0.99, 'batch_episodes': 32, 'epochs': 10, 'GAE': 0.95,
                        'epsilon_clip': 0.25,
                        'value_clip': None, 'grad_clip': None, 'entropy_weight': 0.01, 'hidden_dims': [400, 200, 200],
                        'curiosity_hidden_dim': 128},
            "DDPG": {'actor_lr': 0.0001, 'critic_lr': 0.001, 'batch_size': 100, 'min_playback': 0,
                     'layer_dims': [400, 200], 'tau': 0.001, "update_freq": 1, 'learn_freq': 1},
            "TD3": {'actor_lr': 0.00025, 'critic_lr': 0.00025}
            # , "exploration_steps":5000, "min_memory_for_learning":10000, "batch_size": 256}
        }

    elif env_name == "PongNoFrameskip-v4":
        agent_configs = {
            "DQN": {'lr': 0.0001, "min_playback": 1000, "max_playback": 100000, "update_freq": 1000,
                    'hiden_layer_size': 512, "normalize_state": True, 'epsilon_decay': 30000},
            "PPO": {'lr': 0.0001, 'batch_episodes': 8, 'epochs': 4, 'GAE': 1.0, 'epsilon_clip': 0.2, 'value_clip': None,
                    'grad_clip': None, 'entropy_weight': 0.01, 'hidden_dims': [400, 200]},
        }


    elif env_name == "BreakoutNoFrameskip-v4" or env_name == "BreakoutDeterministic-v4":
        agent_configs = {
            "DQN": {'lr': 0.00001, "min_playback": 50000, "max_playback": 1000000, "update_freq": 10000,
                    'learn_freq': 4,
                    "normalize_state": True, 'epsilon_decay': 5000000},
            "PPO": {'lr': 0.00025, 'lr_decay': 0.9999, 'batch_episodes': 4, 'epochs': 3, 'minibatch_size': 32,
                    'GAE': 0.95, 'epsilon_clip': 0.1, 'value_clip': None,
                    'grad_clip': 0.5, 'entropy_weight': 0.01, 'fe_layers': [512, 512], 'model_layers': [],
                    'horizon': None},
            "PPOParallel": {'lr': 0.00025, 'lr_decay': 0.9999, 'concurrent_epsiodes': 16, 'epochs': 3,
                            'minibatch_size': 32, 'GAE': 0.95,
                            'epsilon_clip': 0.1, 'value_clip': None,
                            'grad_clip': 0.5, 'entropy_weight': 0.01, 'fe_layers': [512, 512], 'model_layers': []}
        }

    elif env_name == "HalfCheetahMuJoCoEnv-v0":
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

    elif "SuperMarioBros" in env_name:
        agent_configs = {
            "DQN": {'lr': 0.0001, 'batch_size': 1, 'learn_freq': 9999999, "min_playback": 1000, "max_playback": 100000,
                    "update_freq": 1000, 'hiden_layer_size': 16, "normalize_state": True, 'epsilon_decay': 30000},
            "PPO": {'lr': 0.0001, 'batch_episodes': 8, 'epochs': 4, 'GAE': 1.0, 'epsilon_clip': 0.2, 'value_clip': None,
                    'grad_clip': None, 'entropy_weight': 0.01, 'hidden_dims': [400, 200]},
        }

    else:
        agent_configs = {}

    return agent_configs[agent_name]