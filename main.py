import gym
from agents.DQN_agent import *
from agents.policy_gradient_agent import *
from agents.advanced_baseline_pg_agent import *
import numpy as np
from time import time, sleep
from torch.utils.tensorboard import SummaryWriter
from _collections import deque

def train(env, actor, train_episodes):
    writer_1 = SummaryWriter(log_dir=os.path.join(TRAIN_DIR, "tensorboard_outputs",  actor.name))
    num_steps = 0
    total_rewards = deque(maxlen=100)
    for i in range(train_episodes):
        episode_start = time()
        done = False
        prev_state = None
        prev_action = None
        reward = 0
        cur_state = env.reset()
        episode_rewards = []
        act_times = []
        action_stats = np.zeros(a)
        while not done:
            start = time()
            action = actor.process_new_state(prev_state, prev_action, reward, cur_state.copy(), done)
            action_stats[action] += 1
            act_times += [time() - start]

            prev_state = cur_state
            prev_action = action

            cur_state, reward, done, info = env.step(action)

            num_steps+=1
            episode_rewards += [reward]
        episode_score = np.sum(episode_rewards)
        total_rewards.append(episode_score)
        actor.process_new_state(prev_state, prev_action, reward, cur_state.copy(), done)
        writer_1.add_scalar('avg_rewards', torch.tensor(np.mean(episode_rewards)), global_step=i)
        writer_1.add_scalar('episode_score', torch.tensor(episode_score), global_step=i)
        writer_1.add_scalar('last_100_episodes_avg', torch.tensor(np.mean(total_rewards)), global_step=i)
        writer_1.add_scalar('episode_length', len(episode_rewards), global_step=i)
        print('Episode',i)
        print("\t# Step", num_steps)
        print("\t# act_time", np.mean(act_times))
        print("\t# steps/sec", len(episode_rewards)/(time()-episode_start))
        print("\t# act_stats",  action_stats / np.sum(action_stats))
        print("\t# Agent stats: ", actor.get_stats())

    actor.save_state(os.path.join(TRAIN_DIR, actor.name + "_trained_weights.pt"))

    env.close()

def test(env,  actor):
    actor.load_state(os.path.join(TRAIN_DIR, actor.name + "_trained_weights.pt"))
    done = False
    cur_state = env.reset()
    prev_state = None
    prev_action = None
    reward = 0
    total_reward = 0
    i = 0
    while not done:
        i+=1
        env.render()
        # sleep(0.1)
        action = actor.process_new_state(prev_state, prev_action, reward, cur_state.copy(), done)
        prev_state = cur_state
        prev_action = action
        cur_state, reward, done, info = env.step(action)
        total_reward += reward
        print(action)
    print("total reward: %f, # steps %d"%(total_reward,i))
    env.close()

if  __name__ == '__main__':
    # ENV_NAME="CartPole-v1"; s=4; a=2
    ENV_NAME="LunarLander-v2"; s=8; a=4
    # ENV_NAME="BipedalWalker-v3"; s=24; a=4
    os.makedirs("Training", exist_ok=True)
    TRAIN_DIR = os.path.join("Training", ENV_NAME)
    os.makedirs(TRAIN_DIR, exist_ok=True)

    env = gym.make(ENV_NAME)
    # env.seed(0)
    NUM_EPISODES = 1000
    actor = DQN_agent(s, a, NUM_EPISODES, train=True)
    # actor = policy_gradient_agent(s, a, NUM_EPISODES, train=True)
    # actor = actor_critic_agent(s, a, NUM_EPISODES, train=True)

    train(env, actor, NUM_EPISODES)

    # actor.train = False
    # actor.epsilon = 0.0
    # test(env, actor)