from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import setup_project_root

setup_project_root()

from src.dialogue.tutoring import run_tutoring
from src.utils.config import load_config
from src.utils.io import read_json
from src.utils.random import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-base-url", default=None)
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--student-preset", default=None)
    args = parser.parse_args()

    config = load_config(args.config, student_preset=args.student_preset)
    seed_everything(args.seed if args.seed is not None else config["random_seed"])
    manifest = read_json(Path(config["split_data_dir"]) / "splits_manifest.json")
    run_tutoring(
        config=config,
        student_checkpoint=args.student_checkpoint,
        tutoring_path=manifest["finqa_tutor_train"],
        output_dir=args.output_dir,
        teacher_base_url=args.teacher_base_url or config["teacher_vllm_base_url"],
        teacher_model=args.teacher_model or config["teacher_model_name"],
    )


if __name__ == "__main__":
    main()