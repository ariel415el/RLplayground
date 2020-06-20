import gym
import os
import numpy as np

def run_episode(env, agent):
    episode_rewards = []
    done = False
    state = env.reset()
    # lives = env.unwrapped.ale.lives()
    while not done:
        action = agent.process_new_state(state)
        state, reward, done, info = env.step(action)
        is_terminal = done
        # cur_life = env.unwrapped.ale.lives()
        # if cur_life < lives:
        #     is_terminal = True
        #     lives = cur_life
        # if hasattr(env, '_max_episode_steps'):
        #     is_terminal = done and len(episode_rewards) < env._max_episode_steps
        agent.process_output(state, reward, is_terminal)
        episode_rewards += [reward]
    return episode_rewards

def train_agent(env, agent, train_dir, logger, solved_score, test_frequency, train_episodes, test_episodes, save_videos, checkpoint_steps):
    next_progress_checkpoint = 1
    next_test_progress_checkpoint = 1

    first_test = test(env, agent, test_episodes)
    logger.log_test(first_test)
    for i in range(train_episodes):

        episode_rewards = run_episode(env, agent)

        # Test model
        if (i+1) % test_frequency == 0:
            if save_videos:
                env = gym.wrappers.Monitor(env, os.path.join(train_dir, "test_%d" % (i+1)), video_callable=lambda episode_id: True, force=True)
            last_test_score = test(env, agent, test_episodes)
            logger.log_test(last_test_score)
            if last_test_score >= next_test_progress_checkpoint * checkpoint_steps * solved_score:
                agent.save_state(os.path.join(train_dir, "test_%.5f_weights.pt" % last_test_score))
                next_test_progress_checkpoint += 1

        # update logger
        logger.update_train_episode(episode_rewards)
        last_k_scores = logger.get_last_k_episodes_mean()

        # output stats
        if last_k_scores >= next_progress_checkpoint * checkpoint_steps * solved_score:
            agent.save_state(os.path.join(train_dir, agent.name + "_%.5f_weights.pt" % last_k_scores))
            next_progress_checkpoint += 1

        if last_k_scores > solved_score:
            print("Solved in %d episodes" % i)
            break

    logger.pickle_episode_scores()
    env.close()

def test(env,  actor, test_episodes=1, render=False, delay=0.0):
    actor.train = False
    episodes_total_rewards = []
    for i in range(test_episodes):
        done = False
        state = env.reset()
        all_rewards = []
        while not done:
            if render:
                env.render()
            action = actor.process_new_state(state)
            state, reward, done, info = env.step(action)
            all_rewards += [reward]

        episodes_total_rewards += [np.sum(all_rewards)]
    score = np.mean(episodes_total_rewards)
    env.close()
    actor.train=True
    return score