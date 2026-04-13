from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import setup_project_root

setup_project_root()

from src.data.prepare import prepare_splits
from src.utils.config import load_config
from src.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--student-preset", default=None)
    args = parser.parse_args()

    config = load_config(args.config, student_preset=args.student_preset)
    split_dir = Path(args.split_dir or config["split_data_dir"])
    prepare_splits(
        raw_dir=args.raw_dir or config["raw_data_dir"],
        split_dir=split_dir,
        config=config,
        seed=args.seed if args.seed is not None else config["random_seed"],
    )
    summary = read_json(split_dir / "splits_summary.json")
    for name, stats in summary.items():
        print(f"{name}: requested={stats['requested']} actual={stats['actual']}")


if __name__ == "__main__":
    main()