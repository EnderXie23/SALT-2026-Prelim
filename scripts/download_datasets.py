from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import setup_project_root

setup_project_root()

from src.data.download import download_dataset_from_config
from src.utils.config import load_config
from src.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dataset", default="all", choices=["finqa", "mbpp", "humaneval", "all"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--student-preset", default=None)
    args = parser.parse_args()

    config = load_config(args.config, student_preset=args.student_preset)
    out_dir = Path(args.out_dir or config["raw_data_dir"])
    datasets = [args.dataset] if args.dataset != "all" else ["finqa", "mbpp", "humaneval"]

    manifest = {}
    for name in datasets:
        manifest[name] = download_dataset_from_config(name, config["dataset_sources"][name], out_dir, force=args.force)
    write_json(out_dir / "download_manifest.json", manifest)


if __name__ == "__main__":
    main()