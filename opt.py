SEED = 0
TRAIN_ROOT = "TEST_TRAINING"
SCORE_SCOPE = 100
LOG_FREQUENCY = 10
CKP_STEP=0.2
TRAIN_EPISODES=10**7
TEMPORAL_FREQ=60*30 # every 30 minutes
TEST_FREQ=500
TEST_EPISODES=1
SAVE_VIDEOS=True
ENV_NAME="BreakoutNoFrameskip-v4"
# AGENT_NAME='PPOParallel'
AGENT_NAME='PPO'
LOGGER_TYPE='tensorboard'
# LOGGER_TYPE='plt'
TRAIN=True
WEIGHTS_FILE=None