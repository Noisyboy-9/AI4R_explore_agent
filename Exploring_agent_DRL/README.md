## AIAR RL practical assignment: 2D exploring robot

This folder contains the code for the 2D exploration practical. The default task is a multi-room checkpoint coverage challenge trained with PPO. The rooms map is scaled larger than the original racing-track exercise so the robot has clearer visual clearance from walls during rollout inspection.

## Platform support

| Platform | CPU training | NVIDIA GPU training | Notes |
| --- | --- | --- | --- |
| Windows 10/11 | Supported | Supported with NVIDIA GPU and working drivers | Use Anaconda Prompt or Miniconda Prompt. |
| Ubuntu/Linux | Supported | Supported with NVIDIA GPU and working drivers | Suitable for desktop or lab machines. |
| macOS Intel or Apple Silicon | Supported in CPU mode | Not supported through CUDA | Apple Metal/MPS is not configured for this exercise. |

Use the CPU environment unless an NVIDIA GPU is available and `nvidia-smi` works.

## Get the code

Clone the repository, then enter this assignment folder:

```bash
git clone https://github.com/UAV-Centre-ITC/AI4R_explore_agent.git
cd AI4R_explore_agent/Exploring_agent_DRL
```

If the repository path contains spaces, wrap the path in quotes when using `cd`.

## Environment files

Two Conda environment files are provided:

- CPU-only: `environment.yml`, environment name `aiar-rl-explore`
- NVIDIA GPU: `environment-gpu.yml`, environment name `aiar-rl-explore-gpu`

Both environments install the local `Explore-Agent` package in editable mode.

## Windows setup with Conda

Open **Anaconda Prompt** or **Miniconda Prompt**, then change to the assignment folder:

```bat
cd C:\path\to\AI4R_explore_agent\Exploring_agent_DRL
```

If the path contains spaces:

```bat
cd "C:\path with spaces\AI4R_explore_agent\Exploring_agent_DRL"
```

### Windows CPU-only setup

Use this if CUDA is not needed or if the computer does not have an NVIDIA GPU:

```bat
set PYTHONNOUSERSITE=1
conda env create -f environment.yml
conda activate aiar-rl-explore
```

If the environment already exists and the dependency file changed:

```bat
set PYTHONNOUSERSITE=1
conda env update -f environment.yml --prune
conda activate aiar-rl-explore
```

### Windows NVIDIA GPU setup

Use this only if the machine has an NVIDIA GPU and this command works:

```bat
nvidia-smi
```

Create and activate the GPU environment:

```bat
set PYTHONNOUSERSITE=1
conda env create -f environment-gpu.yml
conda activate aiar-rl-explore-gpu
```

Verify CUDA before training:

```bat
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

Only use `--num-gpus 1` after `torch.cuda.is_available()` prints `True`.

## Ubuntu/Linux setup with Conda

Open a terminal, then change to the assignment folder:

```bash
cd /path/to/AI4R_explore_agent/Exploring_agent_DRL
```

### Ubuntu/Linux CPU-only setup

```bash
export PYTHONNOUSERSITE=1
conda env create -f environment.yml
conda activate aiar-rl-explore
```

If the environment already exists and the dependency file changed:

```bash
export PYTHONNOUSERSITE=1
conda env update -f environment.yml --prune
conda activate aiar-rl-explore
```

### Ubuntu/Linux NVIDIA GPU setup

Use this only if the machine has an NVIDIA GPU and this command works:

```bash
nvidia-smi
```

Create and activate the GPU environment:

```bash
export PYTHONNOUSERSITE=1
conda env create -f environment-gpu.yml
conda activate aiar-rl-explore-gpu
```

Verify CUDA before training:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

Only use `--num-gpus 1` after `torch.cuda.is_available()` prints `True`.

For remote/headless Ubuntu machines, training can run without a display because the training script uses `gui=False`. Rollout and manual driving need a graphical display for Pygame.

## macOS setup with Conda

Use macOS in CPU-only mode. The provided CUDA GPU environment is not for macOS.

First try the standard CPU environment:

```bash
export PYTHONNOUSERSITE=1
conda env create -f environment.yml
conda activate aiar-rl-explore
```

If Conda cannot solve the environment on macOS, create a CPU environment manually:

```bash
export PYTHONNOUSERSITE=1
conda create -n aiar-rl-explore python=3.10 pip "setuptools<81" -c conda-forge
conda activate aiar-rl-explore
conda install pytorch=2.0 -c pytorch
pip install -e ./Explore-Agent
pip install tensorboard
```

Then run training with `--num-gpus 0`.

If Ray/RLlib installation fails on Apple Silicon, use a Windows or Ubuntu/Linux machine. This exercise uses Ray RLlib, PyTorch, and Pygame rather than native Apple Metal/MPS training.

## Quick environment check

Run this after activating either `aiar-rl-explore` or `aiar-rl-explore-gpu`.

Windows:

```bat
python -c "from explore_agent.envs.exploring_gym import ExploreDrone; env = ExploreDrone({'gui': False, 'env_name': 'rooms', 'reward_mode': 'coverage', 'max_steps': 5}); obs, _ = env.reset(); print('observation shape:', obs.shape); obs, reward, terminated, truncated, info = env.step([0.0, 0.0]); print(round(float(reward), 4), terminated, truncated, info)"
```

Ubuntu/Linux/macOS:

```bash
python -c "from explore_agent.envs.exploring_gym import ExploreDrone; env = ExploreDrone({'gui': False, 'env_name': 'rooms', 'reward_mode': 'coverage', 'max_steps': 5}); obs, _ = env.reset(); print('observation shape:', obs.shape); obs, reward, terminated, truncated, info = env.step([0.0, 0.0]); print(round(float(reward), 4), terminated, truncated, info)"
```

## Manual driving

Use this first to confirm Pygame rendering and the map work.

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

Run these commands from `Exploring_agent_DRL` after activating the Conda environment.

The default rooms task uses a coverage observation with several nearby unvisited checkpoints, not only one next checkpoint. Each checkpoint slot points to the checkpoint midpoint, which keeps the policy from aiming at endpoints close to walls or already visited gates. PPO is also run with a small entropy coefficient by default, and the environment gives penalties for hovering, repeated wall contact, and not approaching a visible unvisited checkpoint. Wall contact penalties increase within each checkpoint-to-checkpoint segment but are bounded and reset after reaching a new checkpoint. These changes make the task behave more like exploration than the older lap-following example.

CPU-only short run:

```bash
python Explore_PPO_agent_training.py --iterations 20 --num-workers 0 --num-gpus 0 --env-name rooms --reward-mode coverage --max-steps 400
```

CPU-only longer run:

```bash
python Explore_PPO_agent_training.py --iterations 500 --num-workers 0 --num-gpus 0 --env-name rooms --reward-mode coverage --max-steps 400
```

GPU training run, only from the `aiar-rl-explore-gpu` environment on Windows or Ubuntu/Linux:

```bash
python Explore_PPO_agent_training.py --iterations 500 --num-workers 0 --num-gpus 1 --env-name rooms --reward-mode coverage --max-steps 400
```

Checkpoints are written to `tmp/ppo_rooms/`. The best checkpoint is saved at:

```text
tmp/ppo_rooms/checkpoint_best
```

## Test a trained PPO checkpoint

```bash
python explore_agent_rollout.py --checkpoint tmp/ppo_rooms/checkpoint_best --env-name rooms --reward-mode coverage --max-steps 400 --gui
```

The rollout opens the Pygame window and prints the cumulative reward, checkpoint coverage, checkpoint reward maximum, and accumulated shaping penalties when the episode ends. The default `rooms` map has 20 checkpoints worth `0.5` each, so the checkpoint reward maximum is `10.0`. Walls are black, unvisited checkpoints are red, visited checkpoints turn green, and yellow rays show the robot's distance sensors. Rays stop at the nearest visible unvisited checkpoint before a wall and mark that checkpoint hit, so the visual sensor overlay does not pass through active goals. The GUI legend explains the map, sensor rays, ray hits, and motion overlays. Live labels near the robot show acceleration, braking, and turn commands; green/blue arrows pulse for acceleration or braking, orange arcs pulse for turning, and the dark arrow shows the actual velocity direction. The simulator renders the full environment internally and displays it in a smaller `1080 x 675` window for laptop screens.

For a terminal-only check without opening a Pygame window:

```bash
python explore_agent_rollout.py --checkpoint tmp/ppo_rooms/checkpoint_best --env-name rooms --reward-mode coverage --max-steps 400 --steps 400 --sleep 0 --no-gui
```

Ray uses a short temporary directory under the system temp folder by default. This avoids path-length issues when the repository is stored in a deeply nested directory.

## Optional playground baseline

The older lap-following task is still available for comparison:

```bash
python Explore_PPO_agent_training.py --iterations 500 --num-workers 0 --num-gpus 0 --env-name playground --reward-mode continuous --max-steps 1000 --checkpoint-dir tmp/ppo_playground
```

## Visualize training metrics

Ray writes training logs under the user home `ray_results` folder by default.

Windows:

```bat
python -m tensorboard.main --logdir %USERPROFILE%\ray_results
```

Ubuntu/Linux/macOS:

```bash
python -m tensorboard.main --logdir ~/ray_results
```

Open the TensorBoard URL printed in the terminal.
