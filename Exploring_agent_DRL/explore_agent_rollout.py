import argparse
import os
import tempfile
from pathlib import Path
from time import sleep

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray._private import resource_spec
from ray.tune.registry import register_env

from explore_agent.envs.exploring_gym import ExploreDrone

os.environ["RAY_USE_CUSTOM_LOGGING"] = "0"

SELECT_ENV = "ExploreAgent-v0"


# Monkey patch to skip nvidia-smi check
def _autodetect_num_gpus():
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Roll out a trained ExploreAgent checkpoint.")
    parser.add_argument("--checkpoint", default="tmp/ppo_rooms/checkpoint_best")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--env-name", default="rooms", choices=["default", "empty", "level2", "random", "playground", "rooms"])
    parser.add_argument("--reward-mode", default="coverage", choices=["dynamic", "continuous", "static", "coverage"])
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--ray-temp-dir", default=str(Path(tempfile.gettempdir()) / "aiar_ray"))
    gui_group = parser.add_mutually_exclusive_group()
    gui_group.add_argument("--gui", action="store_true", help="Open the Pygame window during rollout.")
    gui_group.add_argument("--no-gui", action="store_true", help="Run rollout without opening the Pygame window.")
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint):
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


def make_env_config(args, gui):
    return {
        "env_name": args.env_name,
        "reward_mode": args.reward_mode,
        "max_steps": args.max_steps,
        "gui": gui,
        "gui_draw_echo_vectors": gui,
        "gui_draw_echo_points": gui,
        "gui_velocity": gui,
    }


def main():
    args = parse_args()
    checkpoint = resolve_checkpoint_path(Path(args.checkpoint))
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Train first with Explore_PPO_agent_training.py."
        )

    resource_spec._autodetect_num_gpus = _autodetect_num_gpus
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        _temp_dir=str(Path(args.ray_temp_dir).resolve()),
    )

    register_env(SELECT_ENV, lambda config: ExploreDrone(make_env_config(args, gui=False)))
    agent = Algorithm.from_checkpoint(str(checkpoint))
    env = ExploreDrone(make_env_config(args, gui=not args.no_gui))
    state, _ = env.reset()

    total_reward = 0.0
    try:
        for step in range(args.steps):
            action = agent.compute_single_action(state, explore=False)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if not args.no_gui:
                env.render()

            if terminated or truncated:
                reason = info.get("done_reason", "time_limit" if truncated else "done")
                coverage = ""
                if "visited_checkpoints" in info:
                    coverage = (
                        f"; checkpoints {info['visited_checkpoints']}/{info['total_checkpoints']} "
                        f"({100 * info['coverage_ratio']:.1f}%); max reward {info['max_reward']:.1f}"
                    )
                print(f"Episode ended after {step + 1} steps ({reason}); cumulative reward {total_reward:.2f}{coverage}")
                break

            if args.sleep > 0:
                sleep(args.sleep)
        else:
            coverage = ""
            if "visited_checkpoints" in info:
                coverage = (
                    f"; checkpoints {info['visited_checkpoints']}/{info['total_checkpoints']} "
                    f"({100 * info['coverage_ratio']:.1f}%); max reward {info['max_reward']:.1f}"
                )
            print(f"Finished {args.steps} rollout steps; cumulative reward {total_reward:.2f}{coverage}")
    finally:
        agent.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
