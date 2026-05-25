# AI4R 2D Checkpoint Exploration Practical Assignment

This repository contains a standalone practical assignment for training a deep reinforcement learning agent to explore a 2D multi-room environment. The robot must cross as many checkpoint gates as possible within the time limit while avoiding walls and explaining the learned behavior.

![Pygame snapshot of the 2D checkpoint exploration task](Exploring_agent_DRL/docs/rooms_task_overview.png)

The figure above is a screenshot from the current Pygame renderer. The yellow observation rays stop at the nearest visible wall or checkpoint intersection; they should not pass through black walls. Red gates are unvisited checkpoints and green gates are already visited checkpoints.

## Assignment Guide

The student handout is available as a PDF:

[AI4R Explore Agent Assignment Guide](Exploring_agent_DRL/docs/AI4R_explore_agent_assignment.pdf)

## Student Assignment

Train a PPO agent for the `2d_checkpoint_exploration` task and report how well it behaves. The maximum checkpoint reward is:

```text
20 checkpoints x 0.5 reward = 10.0
```

Your goal is to get as high a score as possible. A score above `6.0` at any point during the allowed rollout is enough to submit the assignment, but the report must explain why the robot behaves as shown in the rollout video. The final reward may be lower if the robot later collides, stalls, or accumulates penalties. A higher and more stable score is better, especially if the behavior does not rely on repeated wall contact. For the final video and score, the rollout may run for up to `1000` steps.

Start with the provided setting, `entropy_coeff=0.1`, and first try to reach more than `6.0` during a `1000` step rollout. The provided setup has been tested to reach this target with enough training and without major code changes. After that, research what entropy regularization does in PPO, test other entropy values, and explain which value gave the best behavior.

Required experiments:

- Train PPO with the provided setting, `entropy_coeff=0.1`.
- Research how `entropy_coeff` affects PPO exploration.
- Choose and test at least two additional entropy values.
- Compare the reward curves, final checkpoint coverage, wall-contact behavior, and visual rollout behavior.
- Explain which entropy value worked best and why.

PPO is the default supported algorithm for the assignment. Students may try another RL algorithm if they keep the map, checkpoints, and reward definition fixed. If another algorithm is used for the submitted result, the report must explain the selected algorithm and repeat the exploration-parameter comparison using that algorithm's closest equivalent to `entropy_coeff`.

Required deliverable:

- A short video or presentation segment showing the best rollout.
- A short explanation of what the robot learned.
- A comparison of the tested entropy settings, including the provided `0.1` setting, discussed or presented in the video report.
- A discussion of failure cases such as getting stuck, oscillating, missing side checkpoints, or colliding with walls.
- The best score reached at any time within the allowed rollout length, plus the final score if it is different.
- If the reward drops strongly after reaching a good score, a clear explanation of what went wrong and what was tried to improve it.
- A zip file containing the source code used for the run and the checkpoint used to record the submitted video.

## Assignment Rules

The walls, checkpoint gates, and reward definition define the task and must stay fixed. Students should not edit `rooms_layout.py`, `reward_config.py`, the checkpoint coordinates, the wall geometry, or the reward terms for the submitted result.

Students may change the learning setup around the fixed task. Reasonable changes include PPO hyperparameters, training duration, entropy coefficient, network settings, action scaling, observation handling, and other variables that affect how the policy learns without changing the task itself. Any such change must be explained in the report and compared against the assignment setup shown in the video.

Keep submitted code readable. Comments should explain decisions that are not obvious from the code, for example why a hyperparameter was changed. Avoid large unrelated rewrites, temporary debug code, machine-specific paths, and comments that simply restate what the next line of code already says.

## Environment Summary

The robot moves in a continuous 2D map with rooms, walls, doorways, and checkpoint gates. Walls are black, unvisited checkpoints are red, and visited checkpoints turn green. Yellow rays show range observations. The robot receives a checkpoint reward only when it crosses a previously unvisited checkpoint gate.

The environment follows the Gymnasium API:

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
env.render()
env.close()
```

## Code Organization

The assignment code is split so students can read the environment, reward definition, and run scripts separately:

```text
Exploring_agent_DRL/
├── run_assignment.py                         # simple wrapper for check/train/rollout
├── Explore_PPO_agent_training.py             # PPO training script
├── Explore_DDPG_agent_training.py            # optional DDPG training script
├── explore_agent_rollout.py                  # checkpoint rollout and visualization
├── performance_sanity_check.py               # headless stepping/render sanity check
├── docs/
│   ├── AI4R_explore_agent_assignment.pdf
│   ├── rooms_task_overview.png
│   └── original_observation_space.png
├── Explore-Agent/explore_agent/envs/
│   ├── exploring_gym.py                      # Gymnasium environment and robot dynamics
│   ├── rooms_layout.py                       # fixed map and checkpoint layout
│   ├── reward_config.py                      # reward constants and task constants
│   └── start_human.py                        # manual driving/debugging
├── environment.yml                           # CPU Conda environment
└── environment-gpu.yml                       # NVIDIA GPU Conda environment
```

Students should start by reading:

- `reward_config.py` for the reward values.
- `rooms_layout.py` for walls and checkpoints.
- `exploring_gym.py` for observations, dynamics, collisions, and Gymnasium `step()`.
- `Explore_PPO_agent_training.py` for PPO configuration and logging.
- `Explore_DDPG_agent_training.py` only if you want to try an optional deterministic actor-critic baseline.

## Action Space

The policy outputs two continuous actions:

```text
action = [propulsion, rotation]
```

- `propulsion > 0`: accelerate forward.
- `propulsion < 0`: brake/reverse.
- `rotation > 0`: rotate right.
- `rotation < 0`: rotate left.

Both action values are clipped to `[-1, 1]`. The robot has damped motion, limited speed, steering authority, and a wall-clearance collision radius so the visible robot body does not overlap walls.

## Observation Space

The default `coverage` task has observation shape `(27,)`.

It includes:

- `7` range rays around the robot.
- Robot speed.
- Angle between heading and velocity.
- Angle to the currently selected checkpoint target.
- `5` unvisited checkpoint slots.
- Overall checkpoint progress.
- Stall time since the last new checkpoint.

The first `10` values describe the robot's local motion and target direction:

```text
[l1, l2, l3, l4, l5, l6, l7, v, alpha, beta]
```

![Observation-space illustration](Exploring_agent_DRL/docs/original_observation_space.png)

- `l1..l7`: normalized wall-clearance distances from the range rays.
- `v`: normalized robot speed.
- `alpha`: normalized angle between the robot heading and its current velocity direction.
- `beta`: normalized angle between the robot heading and the selected checkpoint target direction.

The remaining observation values describe several visible unvisited checkpoint candidates, so the policy can choose between multiple exploration directions instead of only reacting to one target.

Each checkpoint slot contains:

```text
[relative angle, distance, visible flag]
```

The visible flag is only positive when the checkpoint can be seen without a wall blocking line of sight. If the robot overlaps or is too close to a wall, rays are blocked and checkpoints behind the wall are not visible.

## Reward Function

The default reward mode is:

```text
reward_mode="coverage"
```

Reward and shaping values are defined in:

```text
Exploring_agent_DRL/Explore-Agent/explore_agent/envs/reward_config.py
```

Current reward definition:

```text
+0.5    crossing a new checkpoint for the first time
 0.0    revisiting an already visited checkpoint
-0.004  hovering in already explored space
-0.020  being blocked by a wall
-0.002  not getting closer to a visible unvisited checkpoint
bounded wall-contact penalty, accumulating up to -0.5 between new checkpoints
```

The bounded wall-contact penalty resets after the robot reaches a new checkpoint. The extra `-0.020` blocked-wall penalty prevents policies from pushing into walls after the bounded penalty has saturated.

Checkpoint collection uses a swept crossing test. The robot must move across the checkpoint gate; driving near a checkpoint on the same side does not count.

## Environment Setup

The codebase was developed and tested on Ubuntu/Linux. The commands below use an Ubuntu/Linux shell. Windows and macOS should also work through Conda in theory, but they were not tested end to end for this release.

Platform setup references:

- Ubuntu/Linux Conda install guide: <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>
- Windows Conda install guide: <https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html>
- macOS Conda install guide: <https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html>

This README is the run-command guide for the assignment. The command blocks below were tested end to end on Ubuntu/Linux. On Windows, run the same `conda` and `python` commands from Anaconda Prompt or a Conda-enabled PowerShell. On macOS, run them from Terminal after installing Conda. Shell-specific commands such as `source <path-to-miniconda>/etc/profile.d/conda.sh` apply to Linux/macOS terminals only.

The assumption for every platform is that students can install Conda or Miniconda, create the environment, activate it, enter the `Exploring_agent_DRL` folder, and then run the Python commands from this repository.

This assignment does not require heavy GPU compute. A good CPU is enough for training, although an NVIDIA GPU can be used if CUDA is installed correctly. Use the CPU environment unless `nvidia-smi` works and PyTorch reports CUDA as available.

From the repository root:

```bash
cd Exploring_agent_DRL
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

If `conda` is not available in a normal Linux terminal, initialize it first. Replace `<path-to-miniconda>` with the actual Miniconda or Anaconda installation path on that machine:

```bash
source <path-to-miniconda>/etc/profile.d/conda.sh
conda activate aiar-rl-explore-gpu
```

For a default Miniconda installation in the home folder, this is often:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

Check CUDA before using GPU training:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

Use `--num-gpus 1` only when CUDA is available.

## Quick Checks

Headless environment check:

```bash
python run_assignment.py check --steps 1000
```

Manual driving:

```bash
python Explore-Agent/explore_agent/envs/start_human.py
```

Use the arrow keys to drive. Press `r` to reset.

## Training Commands

Default PPO. Start here:

```text
entropy_coeff = 0.1
```

```bash
python run_assignment.py train --iterations 500 --train-batch-size 2000 --sgd-minibatch-size 256 --num-sgd-iter 10 --num-workers 0 --num-gpus 0 --entropy-coeff 0.1 --checkpoint-dir tmp/ppo_entropy_010
```

GPU version, if CUDA is available:

```bash
python run_assignment.py train --iterations 500 --train-batch-size 2000 --sgd-minibatch-size 256 --num-sgd-iter 10 --num-workers 0 --num-gpus 1 --entropy-coeff 0.1 --checkpoint-dir tmp/ppo_entropy_010
```

You can also call the training script directly:

```bash
python Explore_PPO_agent_training.py --iterations 500 --num-workers 0 --num-gpus 0 --env-name 2d_checkpoint_exploration --reward-mode coverage --max-steps 400 --entropy-coeff 0.1 --train-batch-size 2000 --checkpoint-dir tmp/ppo_entropy_010
```

### Training Length per PPO Iteration

`--iterations` controls how many PPO updates are run. `--train-batch-size` controls how many environment steps are collected before each update. With the default values, training uses roughly:

```text
500 iterations x 2000 environment steps per iteration = 1,000,000 sampled steps
```

Students may change this value if they want longer or shorter PPO updates:

```bash
python run_assignment.py train --iterations 500 --train-batch-size 4000 --num-workers 0 --num-gpus 0 --entropy-coeff 0.1 --checkpoint-dir tmp/ppo_entropy_010_batch4000
```

A larger batch gives PPO a more stable update but each iteration takes longer. A smaller batch gives faster iteration feedback but usually noisier learning.

### PPO Parameters Students Can Tune

Students may adjust the training values to reach the task goal, as long as they keep the map geometry fixed and explain the result.

- `--iterations`: number of collect-and-learn cycles. More iterations usually gives the policy more chances to improve.
- `--train-batch-size`: number of environment steps collected before one PPO update. The default `2000` is about five full `400` step training episodes. Larger values are more stable but slower per iteration.
- `--sgd-minibatch-size`: number of samples used in each optimizer minibatch after the full train batch is collected. It should normally be smaller than `--train-batch-size`.
- `--num-sgd-iter`: number of optimization passes over the collected train batch. Higher values learn more from the same data, but too high can overfit to recent behavior.
- `--entropy-coeff`: entropy coefficient used by PPO. The provided setup uses `0.1`; students should research and compare additional values before choosing the best final setting.
- `--max-steps`: maximum training episode length. The assignment uses `400` during training and allows up to `1000` during final rollout.
- `--num-workers`: number of Ray rollout workers. Keep `0` for simple local training unless the machine has enough CPU cores.
- `--num-gpus`: use `1` only when CUDA is available; otherwise use `0`.

### Optional RL Algorithms

PPO is the recommended route and the provided report commands are written for PPO. Students may use another RL algorithm if they keep `rooms_layout.py`, `reward_config.py`, the checkpoint gates, the walls, and the reward terms unchanged.

For the required exploration comparison, use the closest parameter for the selected algorithm:

| Algorithm | Exploration parameter to compare |
| --- | --- |
| PPO | `entropy_coeff` |
| A3C | `entropy_coeff` in RLlib, although A3C is deprecated in the installed RLlib version |
| SAC | entropy temperature settings, such as `target_entropy` and `initial_alpha`, not PPO-style `entropy_coeff` |
| DDPG | no entropy coefficient; compare action-noise settings such as `--exploration-initial-scale`, `--exploration-final-scale`, and `--exploration-scale-timesteps` |
| TD3 | no entropy coefficient; compare action-noise settings if a TD3 trainer is added |

Optional DDPG run:

```bash
python run_assignment.py train-ddpg --iterations 1000 --num-workers 0 --num-gpus 0 --checkpoint-dir tmp/ddpg_2d_checkpoint_exploration
```

DDPG checkpoints can be tested with the same rollout command by changing the checkpoint path:

```bash
python run_assignment.py rollout --checkpoint tmp/ddpg_2d_checkpoint_exploration/checkpoint_best --steps 1000 --max-steps 1000 --render-fps 30
```

### Continue Training from a Checkpoint

To continue from an existing checkpoint, pass `--resume-from` and write new checkpoints to a new folder:

```bash
python run_assignment.py train --iterations 300 --resume-from tmp/ppo_entropy_010/checkpoint_best --checkpoint-dir tmp/ppo_entropy_010_resume --num-workers 0 --num-gpus 0 --entropy-coeff 0.1 --train-batch-size 2000
```

Use the same fixed map when resuming. If the map, observation format, or reward mode is changed, old checkpoints may not be compatible or may behave differently from the submitted setup.

## Testing a Trained Agent

Visual rollout:

```bash
python run_assignment.py rollout --checkpoint tmp/ppo_entropy_010/checkpoint_best --steps 1000 --max-steps 1000 --render-fps 30
```

Terminal-only rollout:

```bash
python run_assignment.py rollout --checkpoint tmp/ppo_entropy_010/checkpoint_best --steps 1000 --max-steps 1000 --no-gui
```

The rollout prints cumulative reward, checkpoint coverage, maximum checkpoint reward, hover penalty, collision penalty, and progress penalty.

For assessment, use the best score reached at any time within the `1000` step rollout. It does not have to remain above `6.0` until the final frame. However, if the robot reaches a good score and then loses a lot of reward, students should try to troubleshoot the behavior, improve it where possible, and explain the failure clearly after the rollout video.

## What to Discuss in the Report

First report what you learned about entropy regularization in PPO and why you selected the entropy values to test. The provided setup uses `entropy_coeff=0.1`; compare it with at least two additional values and identify the best setting for this task. List any other hyperparameters changed from the provided command, such as `--iterations`, `--train-batch-size`, `--sgd-minibatch-size`, `--num-sgd-iter`, network settings, or other PPO/training options. For each change, explain why it was made and how it affected the score or rollout behavior. If another algorithm is used, report the equivalent algorithm-specific hyperparameters instead.

For each entropy setting tested, discuss:

- best reward during the rollout, final reward, and checkpoint count;
- whether the robot enters multiple rooms;
- whether it slows down enough to cross side checkpoints;
- whether it gets stuck near walls;
- how much wall-contact penalty appears;
- how the behavior changes across the entropy values;
- which entropy setting gave the best rollout and why.

For PPO, use the provided `entropy_coeff=0.1` run as the reference case. The following commands are example additional comparison cases; students may choose different values if they justify them from their entropy research:

```bash
python run_assignment.py train --iterations 500 --train-batch-size 2000 --sgd-minibatch-size 256 --num-sgd-iter 10 --num-workers 0 --num-gpus 0 --entropy-coeff 0.0 --checkpoint-dir tmp/ppo_entropy_000
python run_assignment.py train --iterations 500 --train-batch-size 2000 --sgd-minibatch-size 256 --num-sgd-iter 10 --num-workers 0 --num-gpus 0 --entropy-coeff 0.01 --checkpoint-dir tmp/ppo_entropy_001
```

Test each trained checkpoint, including the provided `0.1` checkpoint, and report what changed in the reward curve and rollout video. If a different RL algorithm is used, run the same kind of comparison with its exploration parameter and explain which parameter was changed.

Keep the complete submission video below about `5` minutes if possible. The video only needs to show the best rollout. Discuss or present the entropy-coefficient results using concise plots, tables, or spoken explanation; there is no need to include a full rollout for every entropy run. If a short clip from another rollout clearly supports a behavior pattern discussed in the report, it can be included briefly.

A good format is: show the best rollout first, then briefly explain the score, the main behavior, any failure near the end, the most important training changes, and the entropy comparison.

Do not report only the reward number. The assignment is about connecting the fixed reward definition, observations, and policy behavior.

## TensorBoard

Ray writes logs under the user `ray_results` folder by default.

Ubuntu/Linux/macOS:

```bash
python -m tensorboard.main --logdir ~/ray_results
```

Windows:

```bat
python -m tensorboard.main --logdir %USERPROFILE%\ray_results
```

Open the TensorBoard URL printed in the terminal and compare reward curves across entropy settings.
