from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import setup_project_root

setup_project_root()

from src.eval.code_eval import evaluate_code
from src.eval.finance_eval import evaluate_finance
from src.utils.config import load_config
from src.utils.io import ensure_dir, read_json, write_json, write_text


def _safe_tag(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def _discover_unlearn_adapters(run_dir: Path) -> list[tuple[str, Path]]:
    variants: list[tuple[str, dict[str, object]]] = []
    for path in sorted(run_dir.iterdir(), key=lambda p: p.name):
        if not path.is_dir() or not path.name.startswith("unlearn"):
            continue
        adapter_path = path / "final_adapter"
        if adapter_path.exists():
            variants.append((path.name, adapter_path))
    return variants

def _discover_tutoring_contexts(run_dir: Path) -> list[tuple[str, dict[str, object]]]:
    variants: list[tuple[str, dict[str, object]]] = []
    for path in sorted(run_dir.iterdir(), key=lambda p: p.name):
        if not path.is_dir() or not path.name.startswith("tutoring"):
            continue
        manifest_path = path / "tutoring_manifest.json"
        tutoring_manifest = read_json(manifest_path) if manifest_path.exists() else {}
        metrics_path = Path(tutoring_manifest.get("teach_ctx_metrics", ""))
        teach_ctx_metrics = read_json(metrics_path) if metrics_path.exists() else {}
        variants.append((path.name, teach_ctx_metrics))
    return variants


def _discover_recover_adapters(run_dir: Path) -> list[tuple[str, Path]]:
    variants: list[tuple[str, Path]] = []
    for path in sorted(run_dir.iterdir(), key=lambda p: p.name):
        if not path.is_dir() or not path.name.startswith("recover"):
            continue
        adapter_path = path / "final_adapter"
        if adapter_path.exists():
            variants.append((path.name, adapter_path))
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--student-preset", default=None)
    args = parser.parse_args()

    config = load_config(args.config, student_preset=args.student_preset)
    run_dir = ensure_dir(args.run_dir)
    manifest = read_json(Path(config["split_data_dir"]) / "splits_manifest.json")

    predictions_dir = ensure_dir(Path(run_dir) / "predictions")
    unlearn_adapter = Path(run_dir) / "unlearn" / "final_adapter"

    metrics: dict[str, object] = {}
    metrics["student_preset"] = config.get("resolved_student_preset")
    metrics["student_model_name_or_path"] = config["student_model_name_or_path"]
    # print(f"Evaluating S0 conditions...", flush=True)
    # metrics["S0"] = {
    #     "finance": evaluate_finance(config, manifest["finqa_test"], predictions_dir / "s0_finance.jsonl"),
    #     "code": evaluate_code(config, manifest["humaneval_test"], predictions_dir / "s0_code", generate_only=args.generate_only),
    # }

    for variant_name, adapter_path in _discover_unlearn_adapters(Path(run_dir)):
        variant_tag = _safe_tag(variant_name)
        condition_name = f"S_{variant_tag}"
        print(f"Evaluating {condition_name} condition...", flush=True)
        metrics[condition_name] = {
            "source_dir": variant_name,
            "finance": evaluate_finance(config, manifest["finqa_test"], predictions_dir / f"{variant_tag}_finance.jsonl", adapter_path=str(adapter_path)),
            "code": evaluate_code(config, manifest["humaneval_test"], predictions_dir / f"{variant_tag}_code", adapter_path=str(adapter_path), generate_only=args.generate_only),
        }

    for variant_name, teach_ctx_metrics in _discover_tutoring_contexts(Path(run_dir)):
        variant_tag = _safe_tag(variant_name)
        condition_name = f"S_unlearn_teach_ctx_{variant_tag}"
        print(f"Evaluating {condition_name} condition...", flush=True)
        metrics[condition_name] = {
            "source_dir": variant_name,
            "finance": teach_ctx_metrics,
        }

    for variant_name, recover_adapter in _discover_recover_adapters(Path(run_dir)):
        variant_tag = _safe_tag(variant_name)
        condition_name = f"S_unlearn_teach_ft_{variant_tag}"
        print(f"Evaluating {condition_name} condition...", flush=True)
        metrics[condition_name] = {
            "source_dir": variant_name,
            "finance": evaluate_finance(
                config,
                manifest["finqa_test"],
                predictions_dir / f"{variant_tag}_finance.jsonl",
                adapter_path=str(recover_adapter),
            ),
            "code": evaluate_code(
                config,
                manifest["humaneval_test"],
                predictions_dir / f"{variant_tag}_code",
                adapter_path=str(recover_adapter),
                generate_only=args.generate_only,
            ),
        }

    finance_drop = metrics["S0"]["finance"]["accuracy"] - metrics["S_unlearn"]["finance"]["accuracy"]
    code_drop = None
    s0_code = metrics["S0"]["code"].get("pass@1")
    su_code = metrics["S_unlearn"]["code"].get("pass@1")
    if s0_code is not None and su_code is not None:
        code_drop = s0_code - su_code

    teach_gain_ctx: dict[str, float] = {}
    teach_gain_ft: dict[str, float] = {}
    unlearn_finance = metrics["S_unlearn"]["finance"]["accuracy"]
    for condition_name, condition_metrics in metrics.items():
        if not condition_name.startswith("S_unlearn_teach_"):
            continue
        if condition_name.startswith("S_unlearn_teach_ctx_"):
            teach_gain_ctx[condition_name] = condition_metrics["finance"].get("gain_over_initial")
        elif condition_name.startswith("S_unlearn_teach_ft_"):
            gain = condition_metrics["finance"]["accuracy"] - unlearn_finance
            teach_gain_ft[condition_name] = gain

    summary = {
        "student_preset": config.get("resolved_student_preset"),
        "student_model_name_or_path": config["student_model_name_or_path"],
        "FinanceDrop": finance_drop,
        "CodeDrop": code_drop,
        "TeachGain_CTX": teach_gain_ctx,
        "TeachGain_FT": teach_gain_ft,
        "conditions": metrics,
    }
    write_json(Path(run_dir) / "metrics.json", summary)

    summary_lines = [
        "# Experiment Summary",
        f"- StudentPreset: {config.get('resolved_student_preset')}",
        f"- StudentModel: {config['student_model_name_or_path']}",
        f"- FinanceDrop: {finance_drop}",
        f"- CodeDrop: {code_drop}",
    ]
    if teach_gain_ctx:
        summary_lines.extend([f"- {name}: {gain}" for name, gain in teach_gain_ctx.items()])
    if teach_gain_ft:
        summary_lines.extend([f"- {name}: {gain}" for name, gain in teach_gain_ft.items()])
    write_text(Path(run_dir) / "summary.md", "\n".join(summary_lines))


if __name__ == "__main__":
    main()
