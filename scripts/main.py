"""
main.py
=======
Shared entry point for all training and evaluation modes.

Usage
-----
  python scripts/main.py                          # train (single env)
  python scripts/main.py --n-envs 4              # train (parallel)
  python scripts/main.py --eval checkpoints/policy_final.pth
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PPO agent: chop wood and craft a crafting table in Minecraft"
    )
    parser.add_argument(
        "--eval",
        metavar="CHECKPOINT",
        help="Path to checkpoint for evaluation (skips training)",
    )
    parser.add_argument(
        "--resume",
        metavar="CHECKPOINT",
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel Minecraft environments (default: 1)",
    )
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=50_000)
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of eval episodes (--eval only)"
    )
    parser.add_argument(
        "--vpt-model", type=str, default="./scripts/foundation-model-1x.model"
    )
    parser.add_argument(
        "--vpt-weights", type=str, default="./scripts/foundation-model-1x.weights"
    )
    parser.add_argument("--print-rewards", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # MineRL/Malmo emit very verbose INFO logs (Java render thread warnings,
    # teleport ambiguity noise, quit attempts).  Cap them at WARNING so they
    # don't drown out training output.
    logging.getLogger("minerl").setLevel(logging.WARNING)

    if args.eval:
        from wood_crafting_agent import evaluate

        evaluate(
            args.eval,
            n_episodes=args.episodes,
            vpt_model=args.vpt_model,
            vpt_weights=args.vpt_weights,
        )
    else:
        from wood_crafting_agent import train

        train(
            total_timesteps=args.timesteps,
            rollout_steps=args.rollout_steps,
            save_dir=args.save_dir,
            save_every=args.save_every,
            resume=args.resume,
            vpt_model=args.vpt_model,
            vpt_weights=args.vpt_weights,
            print_rewards=args.print_rewards,
        )
