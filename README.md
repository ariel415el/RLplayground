# RLplayground
Playing around reinforcement learning algorithms

This is a project acompanies my catching of the recent years developements in the Reinforecement learning domain. It gathers simple implementations of state-of-art RL algorithms under the same API.

# Algorithms Implemented
- DQN: With variants (Double, Dueling PriorityMemory, NoisyNets)
- VanilaPG
- A2C
- PPO: With value and norm clipping variants
- DDPG
- TD3
- ICM: In progress
- Evolution Methods: (ES CEM Etc..) Todo
Agent are sorted by the the action space types they can work with: Discrete, Continuous and Hybrid (Both)

# Accomplishments
## CartPole
- DQN
- PG
## Pendulum
- TD3
## LunarLander
- DQN
## LunarLanderContinuous
- TD3
## BipedalWalker: 
- TD3: <img src=Trained_models/BipedalWalker-v3/TD3_lr%5B0.0003%5D_b%5B256%5D_tau%5B0.0050%5D_uf%5B2%5D/Episode-scores%20(1).png width="400">
- PPO
## BipedalWalkerHardcore
- TD3 fine tuned from agent trained on BipedalWalker)
## AtariPong: 
- DQN: <img src=Trained_models/PongNoFrameskip-v4/DobuleDQN-DuelingDqn-Dqn-lr%5B0.00008%5D_b%5B32%5D_lf%5B1%5D_uf%5B1000%5D/Episode-scores%20(3).png width="400">
- PPO: <img src=Trained_models/PongNoFrameskip-v4/PPO_lr%5B0.0001%5D_b%5B8%5D_GAE%5B1.0%5D/Episode-scores_pong.png width="400">
## AtariBreakout

# Credits
Credit to all those Github repository I aspired from. I consulted a lot of repositories in order while implementing the algorithms, defining architectures and finetuning hyperparameters
- https://github.com/sfujim/TD3.git
- https://github.com/iKintosh/DQN-breakout-Pytorch.git
- https://github.com/Anjum48/rl-examples.git
- https://github.com/chagmgang/pytorch_ppo_rl.git
- https://github.com/shivaverma/OpenAIGym.git
- https://github.com/plopd/deep-reinforcement-learning.git
- https://github.com/adik993/ppo-pytorch.git
- https://github.com/higgsfield/RL-Adventure.git
- https://github.com/Adriel-M/OpenAI-Gym-Solutions.git
- https://github.com/nikhilbarhate99/PPO-PyTorch.git

