import gym
import os
import numpy as np
from time import time
from collections import deque
from Enviroment.MultiEnvs import MultiEnviroment, MultiEnviromentSync

class train_progress_manager(object):
    """This object is responsible of monitoring train progress, logging results"""
    def __init__(self, train_dir, solved_score, score_scope, logger, checkpoint_steps=0.2, train_episodes=1000000, temporal_frequency=60**2):
        self.solved_score = solved_score
        self.train_dir = train_dir
        self.checkpoint_steps = checkpoint_steps
        self.temporal_frequency = temporal_frequency
        self.next_progress_checkpoint = 1
        self.next_time_checkpoint = 1
        self.start_time = time()
        self.logger = logger
        self.score_scope = deque(maxlen=score_scope)
        self.episodes_done = 0
        self.train_episodes = train_episodes
        self.training_complete = False

    def report_episode(self, episode_score, episode_length):
        self.score_scope.append(episode_score)
        score_scope_avg = np.mean(self.score_scope)
        self.logger.log_episode(episode_score, score_scope_avg,  episode_length)

        time_passed = time() - self.start_time
        save_path = None
        if score_scope_avg > self.next_progress_checkpoint * self.checkpoint_steps * self.solved_score:
            save_path = os.path.join(self.train_dir, "progress_ckp-_%.5f.pt" % score_scope_avg)
            self.next_progress_checkpoint += 1
        elif time_passed > self.temporal_frequency * self.next_time_checkpoint:
            save_path =  os.path.join(self.train_dir,"time_ckp_%.3f.pt"%(time_passed/360))
            self.next_time_checkpoint += 1
        self.episodes_done += 1
        if score_scope_avg >= self.solved_score:
            print("Solved in %d episodes" % self.episodes_done)
            self.training_complete = True
        if self.episodes_done >= self.train_episodes:
            self.training_complete = True

        return save_path

    def report_test(self, test_score):
        self.logger.add_costume_log("Test-score", self.episodes_done, test_score)

def train_agent_multi_env(env_builder, agent, progress_manager, test_frequency=250, test_episodes=1, save_videos=False):
    """Train agent that can train with multiEnv objects"""
    multi_env = MultiEnviroment(env_builder, agent.hp['concurrent_epsiodes'])
    # multi_env = MultiEnviromentSync(env_builder, agent.hp['concurrent_epsiodes'])
    total_scores = [0 for _ in range(agent.hp['concurrent_epsiodes'])]
    total_lengths = [0 for _ in range(agent.hp['concurrent_epsiodes'])]
    states = multi_env.get_initial_state()
    while not progress_manager.training_complete:
        actions = agent.process_states(states)
        next_states, rewards, is_next_state_terminals, infos = multi_env.step(actions)
        agent.update_step_results(next_states, rewards, is_next_state_terminals)
        states = next_states

        for i, (reward, done) in enumerate(zip(rewards, is_next_state_terminals)):
            total_scores[i] += reward
            total_lengths[i] += 1
            if done:
                save_path = progress_manager.report_episode(total_scores[i], total_lengths[i])
                if save_path is not None:
                    agent.save_state(save_path)
                total_scores[i] = 0
                total_lengths[i] = 0

                if progress_manager.episodes_done % test_frequency == 0:
                    # Test model
                    if (progress_manager.episodes_done + 1) % test_frequency == 0:
                        test_env = env_builder()
                        if save_videos:
                            test_env = gym.wrappers.Monitor(test_env, os.path.join(progress_manager.train_dir, "test_%d" % (
                                        progress_manager.episodes_done + 1)), video_callable=lambda episode_id: True,
                                                            force=True)
                            test_score = test(test_env, agent, test_episodes)
                        else:
                            test_score = test(test_env, agent, test_episodes)
                        progress_manager.report_test(test_score)
                        test_env.close()

    multi_env.close()


def train_agent(env_generator, agent, progress_manager, test_frequency=250, test_episodes=1, save_videos=False):
    """Train agent on a regular gym enviroment"""
    train_env = env_generator()
    while not progress_manager.training_complete:
        episode_rewards = run_episode(train_env, agent)
        # update logger
        save_path = progress_manager.report_episode(np.sum(episode_rewards), len(episode_rewards))
        if save_path is not None:
            agent.save_state(save_path)

        # Test model
        if (progress_manager.episodes_done+1) % test_frequency == 0:
            test_env = env_generator()
            if save_videos:
                test_env = gym.wrappers.Monitor(test_env, os.path.join(progress_manager.train_dir, "test_%d" % (progress_manager.episodes_done+1)), video_callable=lambda episode_id: True, force=True)
                test_score = test(test_env, agent, test_episodes)
            else:
                test_score = test(test_env, agent, test_episodes)
            progress_manager.report_test(test_score)
            test_env.close()

    train_env.close()


def run_episode(env, agent, render=False):
    """Runs a full episode of a regular gym enviroment"""
    done = False
    state = env.reset()
    episode_rewards = []
    while not done:
        if render:
            env.render()
        action = agent.process_new_state(state)
        state, reward, done, info = env.step(action)
        is_terminal = done
        agent.process_output(state, reward, is_terminal)
        episode_rewards += [reward]
    return episode_rewards


def test(env,  actor, test_episodes=1, render=False):
    episodes_total_rewards = []
    for i in range(test_episodes):
        episodes_total_rewards += [np.sum(run_episode(env, actor,render))]
    score = np.mean(episodes_total_rewards)
    return score