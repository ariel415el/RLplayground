import gym
from Enviroment.EnvWrappers import get_atari_env, get_super_mario_env, get_grid_maze_env

"""Holds the target score of each available enviroment"""
env_goals = {"CartPole-v1":195, "Acrobot-v1":-80, "MountainCar-v0":-110, "Pendulum-v0":-200, "LunarLander-v2":200,
             "LunarLanderContinuous-v2": 200, "BipedalWalker-v3":300, "BipedalWalkerHardcore-v3":300,
             "PongNoFrameskip-v4":20, "BreakoutNoFrameskip-v4":60, "FreewayNoFrameskip-v4":20,
             'AntPyBulletEnv-v0':6000, "Walker2DMuJoCoEnv-v0":6000, 'HumanoidMuJoCoEnv-v0':6000, 'HalfCheetahMuJoCoEnv-v0':6000,
             'SuperMarioBros-1':5000, 'SuperMarioBros-v2':5000, 'SuperMarioBros-v3':5000,
             "MiniGrid-FourRooms-v0":10}

class env_builder(object):
    """Make a factory from the function and arguments that creates an enviroment"""
    def __init__(self,constructor, train_args, test_args=None):
        self.constructor = constructor
        self.train_args = train_args
        self.test_args = train_args if test_args is None else test_args
    def __call__(self, test_config=False):
        if test_config:
            return self.constructor(**self.test_args)
        else:
            return self.constructor(**self.train_args)

# class EnvProcess(object)

def get_env_builder(env_name):
    if env_name == "PongNoFrameskip-v4":
        return env_builder(get_atari_env, {'env_name':env_name, 'frame_stack':1})
    elif env_name == "BreakoutNoFrameskip-v4" or env_name == "FreewayNoFrameskip-v4":
        train_args = {'env_name':env_name,'frame_stack':4, 'use_lazy_frames':False, 'clip_rewards': False,
                                           'episode_life':True, 'no_op_reset':False, 'disable_noop':True}
        test_args = train_args.copy()
        test_args['episode_life'] = False
        return env_builder(get_atari_env, train_args, test_args)
    elif "SuperMarioBros" in env_name:
        return env_builder(get_super_mario_env, {'env_name':env_name})
    elif "MiniGrid" in env_name:
        return env_builder(get_grid_maze_env, {'env_name':env_name, 'image_obs':False})
    else:
        return env_builder(gym.make, {'id':env_name})

def get_env_goal(env_name):
    return env_goals[env_name]