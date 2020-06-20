from matplotlib import pyplot as plt
import sys
import os
import pickle
import numpy as np
def plot(k=200):
    dir_path = sys.argv[1]
    env_name = os.path.basename(dir_path)
    for agent_name in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, agent_name)):
            scores_file = open(os.path.join(dir_path, agent_name, "episode_scores.pkl"), 'rb')
            episode_scores = pickle.load(scores_file)
            avg_episode_scores = []
            for i in range(len(episode_scores)):
                avg_episode_scores += [np.mean(episode_scores[max(0,i-k):i])]
            plt.plot(avg_episode_scores, label=agent_name)
    plt.xlabel('# Episode')
    plt.ylabel("last-%d avg score"%k)
    plt.title(env_name)
    plt.legend(fontsize=7)
    plt.savefig(os.path.join(dir_path,"Score_plot.png"))

if __name__ == '__main__':
    plot(k=200)