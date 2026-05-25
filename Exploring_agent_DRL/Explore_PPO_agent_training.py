import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import ray
import torch

from ray.tune.registry import register_env
from explore_agent.envs.exploring_gym import ExploreDrone
from explore_agent.envs.reward_config import (
    COVERAGE_BLOCKED_WALL_PENALTY,
    COVERAGE_HOVER_PENALTY,
    COVERAGE_PROGRESS_PENALTY,
)
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray._private import resource_spec


# Avoid Ray probing nvidia-smi during CPU-only training.
def _autodetect_num_gpus():
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on the ExploreAgent Gymnasium environment.")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-gpus", type=float, default=0)
    parser.add_argument(
        "--env-name",
        default="2d_checkpoint_exploration",
        choices=["default", "empty", "level2", "random", "playground", "rooms", "2d_checkpoint_exploration"],
    )
    parser.add_argument("--reward-mode", default="coverage", choices=["dynamic", "continuous", "static", "coverage"])
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--checkpoint-dir", default="tmp/ppo_2d_checkpoint_exploration")
    parser.add_argument("--resume-from", default="", help="Optional RLlib checkpoint path to continue training from.")
    parser.add_argument("--ray-temp-dir", default=str(Path(tempfile.gettempdir()) / "aiar_ray"))
    parser.add_argument(
        "--layout-path",
        "--rooms-layout-path",
        dest="rooms_layout_path",
        default="",
        help="Optional .npz layout exported by the local layout editor.",
    )
    parser.add_argument("--entropy-coeff", type=float, default=0.1)
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=2000,
        help="Environment steps collected before each PPO update.",
    )
    parser.add_argument("--sgd-minibatch-size", type=int, default=256)
    parser.add_argument("--num-sgd-iter", type=int, default=10)
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint):
    checkpoint = Path(checkpoint)
    if (checkpoint / "rllib_checkpoint.json").exists():
        return checkpoint

    candidates = sorted(
        checkpoint.glob("checkpoint_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "rllib_checkpoint.json").exists():
            return candidate

    return checkpoint


def get_task_limits(args):
    env = ExploreDrone({
        "env_name": args.env_name,
        "reward_mode": args.reward_mode,
        "max_steps": args.max_steps,
        "gui": False,
        "render_mode": None,
        "rooms_layout_path": args.rooms_layout_path,
    })
    try:
        if args.reward_mode == "coverage":
            return {
                "n_checkpoints": env.env.n_goals,
                "checkpoint_reward": env.drone.COVERAGE_REWARD,
                "max_reward": env.env.n_goals * env.drone.COVERAGE_REWARD,
                "observation_shape": env.observation_space.shape,
                "checkpoint_layout_source": env.env.rooms_layout_source_path or "built-in",
            }
        return {
            "n_checkpoints": None,
            "checkpoint_reward": None,
            "max_reward": None,
            "observation_shape": env.observation_space.shape,
            "checkpoint_layout_source": env.env.rooms_layout_source_path or "built-in",
        }
    finally:
        env.close()


def print_training_context(args, task_limits):
    print("\nTraining PPO")
    print(f"  environment: {args.env_name}")
    print(f"  reward mode: {args.reward_mode}")
    print(f"  max steps per episode: {args.max_steps}")
    print(f"  rollout workers: {args.num_workers}")
    print(f"  requested GPUs: {args.num_gpus}")
    print(f"  observation shape: {task_limits['observation_shape']}")
    print(f"  checkpoint layout: {task_limits['checkpoint_layout_source']}")
    print(f"  entropy coefficient: {args.entropy_coeff}")
    if args.resume_from:
        print(f"  resume from: {args.resume_from}")
    print(
        "  PPO update: "
        f"train batch {args.train_batch_size}, minibatch {args.sgd_minibatch_size}, "
        f"SGD passes {args.num_sgd_iter}"
    )
    if task_limits["max_reward"] is not None:
        print(
            "  coverage task: "
            f"{task_limits['n_checkpoints']} checkpoints x "
            f"{task_limits['checkpoint_reward']:.2f} reward = "
            f"{task_limits['max_reward']:.2f} max reward"
        )
        print(
            "  exploration shaping: "
            f"-{COVERAGE_HOVER_PENALTY:.3f}/step for hovering, "
            f"-{COVERAGE_BLOCKED_WALL_PENALTY:.3f}/step when blocked by a wall, "
            f"-{COVERAGE_PROGRESS_PENALTY:.3f}/step for no progress, "
            "bounded wall-contact penalty"
        )

    print("\nLog columns")
    print("  iter: completed PPO training iteration")
    print("  reward min/mean/max: episode return statistics from the latest training batch")
    print("  train batch: environment steps collected before one PPO update")
    if task_limits["max_reward"] is not None:
        print("  score%: mean reward as a percentage of the checkpoint reward maximum")
        print("          idle penalties can reduce returns below the checkpoint score")
        print("          early scores can be negative because collision has a penalty")
    print("  len: mean episode length in environment steps")
    print("  best: best mean reward saved after warmup iterations\n")
    sys.stdout.flush()


def format_training_status(iteration, result, best_reward, max_reward):
    min_reward = result["episode_reward_min"]
    mean_reward = result["episode_reward_mean"]
    max_seen_reward = result["episode_reward_max"]
    mean_len = result["episode_len_mean"]
    best = best_reward if best_reward != float("-inf") else 0.0

    if max_reward is not None and max_reward > 0:
        score_percent = 100.0 * mean_reward / max_reward
        return (
            f"iter {iteration:4d} | "
            f"reward {min_reward:7.2f}/{mean_reward:7.2f}/{max_seen_reward:7.2f} | "
            f"score {score_percent:6.1f}% | "
            f"len {mean_len:6.1f} | "
            f"best {best:7.2f}"
        )

    return (
        f"iter {iteration:4d} | "
        f"reward {min_reward:7.2f}/{mean_reward:7.2f}/{max_seen_reward:7.2f} | "
        f"len {mean_len:6.1f} | "
        f"best {best:7.2f}"
    )


def main():
    args = parse_args()
    if args.num_gpus > 0 and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU training was requested with --num-gpus > 0, but this environment has CPU-only PyTorch. "
            "Install a CUDA-enabled PyTorch build or run with --num-gpus 0."
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    task_limits = get_task_limits(args)
    print_training_context(args, task_limits)

    if args.num_gpus == 0:
        resource_spec._autodetect_num_gpus = _autodetect_num_gpus
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        _temp_dir=str(Path(args.ray_temp_dir).resolve()),
    )

    select_env = "ExploreAgent-v0"
    register_env(
        select_env,
        lambda config: ExploreDrone({
            "env_name": args.env_name,
            "reward_mode": args.reward_mode,
            "max_steps": args.max_steps,
            "gui": False,
            "render_mode": None,
            "rooms_layout_path": args.rooms_layout_path,
        }),
    )
    config = (
        PPOConfig()
        .resources(num_gpus=args.num_gpus)
        .rollouts(num_rollout_workers=args.num_workers)
        .training(
            entropy_coeff=args.entropy_coeff,
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
        )
        .framework("torch")
    )

    agent = config.build(env=select_env)
    if args.resume_from:
        resume_checkpoint = resolve_checkpoint_path(args.resume_from)
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        agent.restore(str(resume_checkpoint))
        print(f"Restored training state from {resume_checkpoint}")

    best_reward = float("-inf")
    best_checkpoint = None

    try:
        for n in range(args.iterations):
            result = agent.train()
            mean_reward = result["episode_reward_mean"]

            if n >= args.warmup_iterations and mean_reward > best_reward:
                best_reward = mean_reward
                best_path = checkpoint_dir / "checkpoint_best"
                shutil.rmtree(best_path, ignore_errors=True)
                best_checkpoint = agent.save(str(best_path))
                print(f"Iteration {n + 1}: new best mean reward {mean_reward:.2f}")

            if n % args.save_interval == 0:
                agent.save(str(checkpoint_dir / f"checkpoint_{n:04d}"))

            print(format_training_status(n + 1, result, best_reward, task_limits["max_reward"]))
    finally:
        agent.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
