# AI4R Explore Agent Practical Assignment

This repository contains the code for the AI for Autonomous Robotics practical on training a deep reinforcement learning agent for a 2D mobile robot exploration task.

The exercise uses a custom Gymnasium environment, Ray RLlib, PyTorch, and Pygame. The provided training script uses PPO, but students may experiment with other RLlib algorithms.

![2D exploration environment](Exploring_agent_DRL/imgs/space_wp3.jpg)

## Assignment Overview

The goal is to train a simulated mobile robot to explore the `playground` map while avoiding walls and obstacles. A successful policy should drive around the environment and complete a full lap of the map layout.

Students should use the exercise to connect reinforcement learning concepts from the lectures to an implemented DRL workflow:

- reading the environment and agent code;
- understanding the action space, observation space, reward function, and training loop;
- training a DRL policy in the custom environment;
- testing the trained policy visually;
- changing selected hyperparameters and explaining their effect on training and rollout behavior.

The original assignment handout is included here:

- [AIAR_RL_practical_assignment_1-4-a.pdf](AIAR_RL_practical_assignment_1-4-a.pdf)

## Learning Objectives

After completing the practical, students should be able to:

- explain how a custom robot environment is represented using the Gymnasium API;
- describe how observations, actions, rewards, and episode termination are implemented in code;
- train and evaluate a DRL agent using Ray RLlib;
- interpret the effect of selected hyperparameters on reward curves and rollout behavior;
- present the final policy behavior using short videos, plots, or screenshots.

## Environment

The robot moves in a 2D continuous environment with walls, obstacles, and goal/checkpoint regions. The environment follows the standard Gymnasium interaction pattern:

- `reset()` initializes an episode;
- `step(action)` applies one action and returns the next observation, reward, and termination flags;
- `render()` visualizes the robot and map using Pygame.

The main environment implementation is:

```text
Exploring_agent_DRL/Explore-Agent/explore_agent/envs/exploring_gym.py
```

## Action Space

The agent controls the robot using two continuous actions:

```text
action = [propulsion, rotation]
```

- `propulsion > 0`: move forward;
- `propulsion < 0`: move backward;
- `rotation > 0`: rotate right;
- `rotation < 0`: rotate left.

Both action values are clipped to the range `[-1, 1]`.

## Observation Space

The observation vector contains range-like clearance measurements and robot motion/goal information:

```text
observation = [l1, l2, l3, l4, l5, l6, l7, v, alpha, beta]
```

where:

- `l1` to `l7` are clearance values measured at different angles around the robot;
- `v` is the robot speed magnitude;
- `alpha` is the angle between the robot heading and its velocity direction;
- `beta` is the angle between the robot heading and the next goal/checkpoint direction.

The observation values are normalized to match the declared Gymnasium observation space.

## Reward

The reward is based on passing goal/checkpoint regions placed along the map. This encourages the robot to explore new parts of the environment instead of simply maximizing speed. Collisions with walls or obstacles terminate the episode.

Students may experiment with reward shaping through the goal/checkpoint positions, but the wall and obstacle layout should remain unchanged when comparing results between groups.

## Student Task

Train a DRL agent to complete the 2D exploration task. PPO is provided as the default algorithm, but other off-the-shelf RLlib algorithms can also be tested.

Recommended experiments:

- train PPO with the default settings;
- vary one or two selected hyperparameters;
- compare reward curves and rollout behavior;
- test whether the trained policy can complete a full lap;
- document failure cases such as collisions, oscillations, or getting stuck;
- keep the original map layout fixed for fair comparison.

The main PPO training script is:

```text
Exploring_agent_DRL/Explore_PPO_agent_training.py
```

The rollout script for testing a trained checkpoint is:

```text
Exploring_agent_DRL/explore_agent_rollout.py
```

## Deliverable

Prepare a short video or presentation segment showing:

- which DRL algorithm was used;
- which hyperparameters or environment settings were changed;
- training evidence such as reward curves or logs;
- visual rollout of the trained robot;
- whether the robot completed the exploration task;
- examples of both successful and challenging test cases.

The assignment handout specifies a short video report rather than a written report.

## Repository Layout

```text
.
├── AIAR_RL_practical_assignment_1-4-a.pdf
├── Exploring_agent_DRL/
│   ├── README.md
│   ├── environment.yml
│   ├── environment-gpu.yml
│   ├── Explore_PPO_agent_training.py
│   ├── Explore_DDPG_agent_training.py
│   ├── explore_agent_rollout.py
│   ├── Explore-Agent/
│   │   └── explore_agent/envs/
│   └── imgs/
```

## Platform Support

| Platform | CPU training | NVIDIA GPU training | Notes |
| --- | --- | --- | --- |
| Windows 10/11 | Supported | Supported with NVIDIA GPU and working drivers | Use Anaconda Prompt or Miniconda Prompt. |
| Ubuntu/Linux | Supported | Supported with NVIDIA GPU and working drivers | Suitable for desktop or lab machines. |
| macOS Intel or Apple Silicon | Supported in CPU mode | Not supported through CUDA | Apple Metal/MPS is not configured for this exercise. |

Use the CPU environment unless an NVIDIA GPU is available and `nvidia-smi` works.

## Setup

Clone the repository and enter the exercise folder:

```bash
git clone https://github.com/UAV-Centre-ITC/AI4R_explore_agent.git
cd AI4R_explore_agent/Exploring_agent_DRL
```

CPU environment:

```bash
conda env create -f environment.yml
conda activate aiar-rl-explore
```

NVIDIA GPU environment:

```bash
conda env create -f environment-gpu.yml
conda activate aiar-rl-explore-gpu
```

Before using the GPU environment, check that CUDA is visible:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Detailed Windows, Ubuntu/Linux, and macOS setup notes are provided in:

- [Exploring_agent_DRL/README.md](Exploring_agent_DRL/README.md)

## Quick Check

Run this after activating the Conda environment:

```bash
python -c "from explore_agent.envs.exploring_gym import ExploreDrone; env = ExploreDrone({'gui': False, 'env_name': 'playground', 'reward_mode': 'continuous', 'max_steps': 5}); obs, _ = env.reset(); print('observation shape:', obs.shape); obs, reward, terminated, truncated, info = env.step([0.0, 0.0]); print(round(float(reward), 4), terminated, truncated, info)"
```

## Manual Driving

Use manual control first to inspect the map and robot behavior.

Windows:

```bat
python Explore-Agent\explore_agent\envs\start_human.py
```

Ubuntu/Linux/macOS:

```bash
python Explore-Agent/explore_agent/envs/start_human.py
```

Use the arrow keys to drive. Press `r` to reset.

## Train PPO

Short CPU run:

```bash
python Explore_PPO_agent_training.py --iterations 20 --num-workers 1 --num-gpus 0
```

Longer CPU run:

```bash
python Explore_PPO_agent_training.py --iterations 500 --num-workers 2 --num-gpus 0
```

GPU run:

```bash
python Explore_PPO_agent_training.py --iterations 500 --num-workers 2 --num-gpus 1
```

Checkpoints are written to:

```text
tmp/ppo/
```

The best checkpoint is saved at:

```text
tmp/ppo/checkpoint_best
```

## Test a Trained Policy

Visual rollout:

```bash
python explore_agent_rollout.py --checkpoint tmp/ppo/checkpoint_best
```

Terminal-only rollout check:

```bash
python explore_agent_rollout.py --checkpoint tmp/ppo/checkpoint_best --steps 20 --sleep 0 --no-gui
```

## TensorBoard

Ray writes training logs under the user `ray_results` folder.

Windows:

```bat
python -m tensorboard.main --logdir %USERPROFILE%\ray_results
```

Ubuntu/Linux/macOS:

```bash
python -m tensorboard.main --logdir ~/ray_results
```

Open the TensorBoard URL printed in the terminal.
