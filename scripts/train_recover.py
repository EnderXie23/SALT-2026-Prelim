from __future__ import annotations

import argparse

from _bootstrap import setup_project_root

setup_project_root()

from src.train.recover_trainer import train_recovery
from src.utils.config import load_config
from src.utils.random import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--tutoring-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--student-preset", default=None)
    args = parser.parse_args()

    config = load_config(args.config, student_preset=args.student_preset)
    seed_everything(args.seed if args.seed is not None else config["random_seed"])
    train_recovery(config, args.student_checkpoint, args.tutoring_data, args.output_dir)


if __name__ == "__main__":
    main()