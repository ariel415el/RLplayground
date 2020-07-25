import os
from utils import loggers
from Agents.AgentConfigs import *
from Agents.AgentBuilder import build_agent
from Enviroment.EnvBuilder import get_env_builder, get_env_goal
import train
from opt import *
import gym

def get_train_function(agent_name):
    if agent_name == "PPOParallel":
        return train.train_agent_multi_env
    else:
        return train.train_agent


def get_logger(logger_type, log_frequency, logdir):
    if logger_type == 'plt':
        constructor = loggers.plt_logger
    elif logger_type == 'tensorboard':
        constructor = loggers.TB_logger
    else:
        constructor = loggers.logger

    return constructor(log_frequency, logdir)


if __name__ == '__main__':
    # random.seed(SEED)
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)

    env_builder = get_env_builder(ENV_NAME)
    hp = get_agent_configs(AGENT_NAME, ENV_NAME)
    agent = build_agent(AGENT_NAME, env_builder(), hp)
    if WEIGHTS_FILE:
        agent.load_state(WEIGHTS_FILE)
    train_dir = os.path.join(TRAIN_ROOT, ENV_NAME,  agent.name)

    if TRAIN:
        logger = get_logger(LOGGER_TYPE, log_frequency=LOG_FREQUENCY, logdir=train_dir)
        agent.set_reporter(logger)
        progress_maneger = train.train_progress_manager(train_dir, get_env_goal(ENV_NAME), SCORE_SCOPE, logger,
                                                        checkpoint_steps=CKP_STEP, train_episodes=TRAIN_EPISODES,
                                                        temporal_frequency=TEMPORAL_FREQ)

        train_function = get_train_function(AGENT_NAME)
        train_function(env_builder, agent, progress_maneger, test_frequency=TEST_FREQ, test_episodes=TEST_EPISODES,
                       save_videos=SAVE_VIDEOS)

    else:
        # Test
        env = env_builder(test_config=True)
        env = gym.wrappers.Monitor(env, os.path.join(train_dir, "test"),
                            video_callable=lambda episode_id: True, force=True)
        score = train.test(env, agent, TEST_EPISODES, render=True)
        print("Avg reward over %d episodes: %f"%(TEST_EPISODES, score))
