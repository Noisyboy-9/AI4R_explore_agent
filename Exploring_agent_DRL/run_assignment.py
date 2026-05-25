import argparse
import subprocess
import sys


def run_command(command):
    print(" ".join(command), flush=True)
    return subprocess.run(command, check=True).returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Convenience runner for the AI4R explore-agent assignment.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Run a quick headless environment check.")
    check.add_argument("--steps", type=int, default=1000)

    train = subparsers.add_parser("train", help="Train PPO on the 2D checkpoint exploration task.")
    train.add_argument("--iterations", type=int, default=500)
    train.add_argument("--entropy-coeff", type=float, default=0.1)
    train.add_argument("--num-gpus", type=float, default=0)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--max-steps", type=int, default=400)
    train.add_argument("--checkpoint-dir", default="tmp/ppo_2d_checkpoint_exploration")
    train.add_argument("--resume-from", default="")
    train.add_argument("--layout-path", "--rooms-layout-path", dest="rooms_layout_path", default="")
    train.add_argument("--train-batch-size", type=int, default=2000)
    train.add_argument("--sgd-minibatch-size", type=int, default=256)
    train.add_argument("--num-sgd-iter", type=int, default=10)

    train_ddpg = subparsers.add_parser("train-ddpg", help="Optional DDPG training on the same task.")
    train_ddpg.add_argument("--iterations", type=int, default=1000)
    train_ddpg.add_argument("--num-gpus", type=float, default=0)
    train_ddpg.add_argument("--num-workers", type=int, default=0)
    train_ddpg.add_argument("--max-steps", type=int, default=400)
    train_ddpg.add_argument("--checkpoint-dir", default="tmp/ddpg_2d_checkpoint_exploration")
    train_ddpg.add_argument("--resume-from", default="")
    train_ddpg.add_argument("--layout-path", "--rooms-layout-path", dest="rooms_layout_path", default="")
    train_ddpg.add_argument("--train-batch-size", type=int, default=256)
    train_ddpg.add_argument("--actor-lr", type=float, default=1e-3)
    train_ddpg.add_argument("--critic-lr", type=float, default=1e-3)
    train_ddpg.add_argument("--exploration-initial-scale", type=float, default=1.0)
    train_ddpg.add_argument("--exploration-final-scale", type=float, default=0.02)
    train_ddpg.add_argument("--exploration-scale-timesteps", type=int, default=10000)

    rollout = subparsers.add_parser("rollout", help="Visualize or score a trained checkpoint.")
    rollout.add_argument("--checkpoint", default="tmp/ppo_2d_checkpoint_exploration/checkpoint_best")
    rollout.add_argument("--max-steps", type=int, default=1000)
    rollout.add_argument("--steps", type=int, default=1000)
    rollout.add_argument("--render-fps", type=int, default=30)
    rollout.add_argument("--no-gui", action="store_true")
    rollout.add_argument("--layout-path", "--rooms-layout-path", dest="rooms_layout_path", default="")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "check":
        return run_command([
            sys.executable,
            "performance_sanity_check.py",
            "--steps",
            str(args.steps),
            "--skip-visual",
        ])
    if args.command == "train":
        command = [
            sys.executable,
            "Explore_PPO_agent_training.py",
            "--iterations",
            str(args.iterations),
            "--num-workers",
            str(args.num_workers),
            "--num-gpus",
            str(args.num_gpus),
            "--env-name",
            "2d_checkpoint_exploration",
            "--reward-mode",
            "coverage",
            "--max-steps",
            str(args.max_steps),
            "--entropy-coeff",
            str(args.entropy_coeff),
            "--checkpoint-dir",
            args.checkpoint_dir,
            "--train-batch-size",
            str(args.train_batch_size),
            "--sgd-minibatch-size",
            str(args.sgd_minibatch_size),
            "--num-sgd-iter",
            str(args.num_sgd_iter),
        ]
        if args.resume_from:
            command.extend(["--resume-from", args.resume_from])
        if args.rooms_layout_path:
            command.extend(["--rooms-layout-path", args.rooms_layout_path])
        return run_command(command)
    if args.command == "train-ddpg":
        command = [
            sys.executable,
            "Explore_DDPG_agent_training.py",
            "--iterations",
            str(args.iterations),
            "--num-workers",
            str(args.num_workers),
            "--num-gpus",
            str(args.num_gpus),
            "--env-name",
            "2d_checkpoint_exploration",
            "--reward-mode",
            "coverage",
            "--max-steps",
            str(args.max_steps),
            "--checkpoint-dir",
            args.checkpoint_dir,
            "--train-batch-size",
            str(args.train_batch_size),
            "--actor-lr",
            str(args.actor_lr),
            "--critic-lr",
            str(args.critic_lr),
            "--exploration-initial-scale",
            str(args.exploration_initial_scale),
            "--exploration-final-scale",
            str(args.exploration_final_scale),
            "--exploration-scale-timesteps",
            str(args.exploration_scale_timesteps),
        ]
        if args.resume_from:
            command.extend(["--resume-from", args.resume_from])
        if args.rooms_layout_path:
            command.extend(["--rooms-layout-path", args.rooms_layout_path])
        return run_command(command)
    if args.command == "rollout":
        command = [
            sys.executable,
            "explore_agent_rollout.py",
            "--checkpoint",
            args.checkpoint,
            "--env-name",
            "2d_checkpoint_exploration",
            "--reward-mode",
            "coverage",
            "--max-steps",
            str(args.max_steps),
            "--steps",
            str(args.steps),
            "--render-fps",
            str(args.render_fps),
        ]
        command.append("--no-gui" if args.no_gui else "--gui")
        if args.no_gui:
            command.extend(["--sleep", "0"])
        if args.rooms_layout_path:
            command.extend(["--rooms-layout-path", args.rooms_layout_path])
        return run_command(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
