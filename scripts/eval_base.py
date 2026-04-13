from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import setup_project_root

setup_project_root()

from src.eval.code_eval import evaluate_code
from src.eval.finance_eval import evaluate_finance
from src.utils.config import load_config
from src.utils.io import ensure_dir, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--student-preset", default=None)
    args = parser.parse_args()

    config = load_config(args.config, student_preset=args.student_preset)
    output_dir = ensure_dir(args.output_dir)
    manifest = read_json(Path(config["split_data_dir"]) / "splits_manifest.json")
    finance = evaluate_finance(config, manifest["finqa_test"], output_dir / "finance_base.jsonl")
    code = evaluate_code(config, manifest["humaneval_test"], output_dir / "code_base")
    write_json(output_dir / "base_metrics.json", {"finance": finance, "code": code, "student_preset": config.get("resolved_student_preset")})


if __name__ == "__main__":
    main()