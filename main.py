import gym
from agents.DQN_agent import *
from agents.vanila_policy_gradient import *
from agents.actor_critic import *
import numpy as np
from time import time, sleep
from torch.utils.tensorboard import SummaryWriter
from _collections import deque
import random

def train(env, actor, train_episodes):
    train_start = time()
    writer_1 = SummaryWriter(log_dir=os.path.join(TRAIN_DIR, "tensorboard_outputs",  actor.name))
    num_steps = 0
    total_rewards = deque(maxlen=100)
    for i in range(train_episodes):
        done = False
        state = env.reset()
        episode_rewards = []
        while not done:
            action = actor.process_new_state(state)
            state, reward, done, info = env.step(action)
            actor.process_output(state, reward, done)
            num_steps+=1
            episode_rewards += [reward]

        episode_score = np.sum(episode_rewards)
        total_rewards.append(episode_score)
        last_100_score = np.mean(total_rewards)
        writer_1.add_scalar('1.last_100_episodes_avg', torch.tensor(last_100_score), global_step=i)
        writer_1.add_scalar('2.episode_score', torch.tensor(episode_score), global_step=i)
        writer_1.add_scalar('3.episode_length', len(episode_rewards), global_step=i)
        writer_1.add_scalar('4.avg_rewards', torch.tensor(np.mean(episode_rewards)), global_step=i)
        cur_time = max(1,int(time() - train_start))
        writer_1.add_scalar('5.episode_score_time_scaled', torch.tensor(episode_score), global_step=cur_time)
        print('Episode',i)
        print("\t# Step %d, time %d mins:"%(num_steps, cur_time/60))
        print("\t# steps/sec", num_steps/cur_time)
        print("\t# Agent stats: ", actor.get_stats())

        if last_100_score >= 200:
            print("Solved..", last_100_score)
            break
    actor.save_state(os.path.join(TRAIN_DIR, actor.name + "_trained_weights.pt"))

    env.close()

def test(env,  actor):
    actor.load_state(os.path.join(TRAIN_DIR, actor.name + "_trained_weights.pt"))
    done = False
    state = env.reset()
    total_reward = 0
    i = 0
    while not done:
        i+=1
        env.render()
        # sleep(0.1)
        action = actor.process_new_state(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
    print("total reward: %f, # steps %d"%(total_reward,i))
    env.close()

if  __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)
    ENV_NAME="CartPole-v1"; s=4; a=2
    # ENV_NAME="LunarLander-v2"; s=8; a=4
    # ENV_NAME="BipedalWalker-v3"; s=24; a=4
    os.makedirs("Training", exist_ok=True)
    TRAIN_DIR = os.path.join("Training", ENV_NAME)
    os.makedirs(TRAIN_DIR, exist_ok=True)

    env = gym.make(ENV_NAME)
    env.seed(0)
    NUM_EPISODES = 1000
    actor = DQN_agent(s, a, NUM_EPISODES, train=True)
    # actor = policy_gradient_agent(s, a, NUM_EPISODES, train=True)
    # actor = actor_critic_agent(s, a, NUM_EPISODES, train=True)

    train(env, actor, NUM_EPISODES)

    # actor.train = False
    # actor.epsilon = 0.0
    # test(env, actor)