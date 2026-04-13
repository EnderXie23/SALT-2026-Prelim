from __future__ import annotations

import argparse
from pathlib import Path
import random

from _bootstrap import setup_project_root

setup_project_root()

from src.eval.code_eval import evaluate_code
from src.eval.finance_eval import evaluate_finance
from src.utils.config import load_config
from src.utils.finance_unlearning import summarize_forget_predictions
from src.utils.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl, write_text


def _safe_tag(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def _unlearn_sort_key(path: Path) -> tuple[int, str]:
    if path.name == "unlearn":
        return (0, path.name)
    return (1, path.name)


def _discover_unlearn_adapters(run_dir: Path) -> list[tuple[str, Path]]:
    variants: list[tuple[str, Path]] = []
    for path in sorted(run_dir.iterdir(), key=_unlearn_sort_key):
        if not path.is_dir() or not path.name.startswith("unlearn"):
            continue
        adapter_path = path / "final_adapter"
        if adapter_path.exists():
            variants.append((path.name, adapter_path))
    return variants


def _extract_code_score(metrics: dict) -> float | None:
    score = metrics.get("pass@1")
    return float(score) if score is not None else None


def _sample_jsonl(input_path: str | Path, output_path: str | Path, sample_size: int, seed: int) -> Path:
    rows = read_jsonl(input_path)
    rng = random.Random(seed)
    if len(rows) > sample_size:
        rows = rng.sample(rows, sample_size)
    write_jsonl(output_path, rows)
    return Path(output_path)


def _augment_forget_metrics(split_path: str | Path, prediction_path: str | Path, metrics: dict, refusal_template: str) -> dict:
    rows = read_jsonl(split_path)
    predictions = read_jsonl(prediction_path)
    rows_by_id = {str(row["id"]): row for row in rows}
    metrics.update(summarize_forget_predictions(rows_by_id, predictions, refusal_template=refusal_template))
    write_json(str(prediction_path).replace(".jsonl", "_metrics.json"), metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--student-preset", default=None)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--skip-finance-test", action="store_true")
    parser.add_argument("--forget-sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--summary-name", default="summary_unlearning.md")
    parser.add_argument("--metrics-name", default="metrics_unlearning.json")
    args = parser.parse_args()

    config = load_config(args.config, student_preset=args.student_preset)
    run_dir = ensure_dir(args.run_dir)
    predictions_dir = ensure_dir(Path(run_dir) / "predictions")
    manifest = read_json(Path(config["split_data_dir"]) / "splits_manifest.json")
    seed = args.seed if args.seed is not None else int(config["random_seed"])
    refusal_template = str(
        config["train_hparams"].get(
            "forget_refusal_template",
            "I cannot determine the required finance-specific calculation method from the available information.",
        )
    )
    sampled_splits_dir = ensure_dir(Path(run_dir) / "sampled_splits")
    forget_eval_path = _sample_jsonl(
        manifest["finqa_forget_train"],
        sampled_splits_dir / f"finqa_forget_train_sample_{args.forget_sample_size}_seed{seed}.jsonl",
        sample_size=args.forget_sample_size,
        seed=seed,
    )

    conditions: dict[str, object] = {
        "student_preset": config.get("resolved_student_preset"),
        "student_model_name_or_path": config["student_model_name_or_path"],
        "finance_test_enabled": not args.skip_finance_test,
        "forget_sample_size": args.forget_sample_size,
        "forget_sample_seed": seed,
        "forget_eval_path": str(forget_eval_path),
    }

    if args.include_base:
        print("Evaluating S0 baseline...", flush=True)
        s0_forget_output = predictions_dir / "s0_finance_forget_train.jsonl"
        s0_metrics = {
            "finance_forget_train": _augment_forget_metrics(
                forget_eval_path,
                s0_forget_output,
                evaluate_finance(
                    config,
                    forget_eval_path,
                    s0_forget_output,
                ),
                refusal_template=refusal_template,
            ),
            "code": evaluate_code(
                config,
                manifest["humaneval_test"],
                predictions_dir / "s0_code",
                generate_only=args.generate_only,
            ),
        }
        if not args.skip_finance_test:
            s0_metrics["finance_test"] = evaluate_finance(
                config,
                manifest["finqa_test"],
                predictions_dir / "s0_finance_test.jsonl",
            )
        conditions["S0"] = s0_metrics

    discovered = _discover_unlearn_adapters(Path(run_dir))
    if not discovered:
        raise FileNotFoundError(f"No unlearn* adapters with final_adapter found under {run_dir}")

    for variant_name, adapter_path in discovered:
        variant_tag = _safe_tag(variant_name)
        condition_name = f"S_{variant_tag}"
        print(f"Evaluating {condition_name}...", flush=True)
        forget_output_path = predictions_dir / f"{variant_tag}_finance_forget_train.jsonl"
        variant_metrics = {
            "source_dir": variant_name,
            "adapter_path": str(adapter_path),
            "finance_forget_train": _augment_forget_metrics(
                forget_eval_path,
                forget_output_path,
                evaluate_finance(
                    config,
                    forget_eval_path,
                    forget_output_path,
                    adapter_path=str(adapter_path),
                ),
                refusal_template=refusal_template,
            ),
            "code": evaluate_code(
                config,
                manifest["humaneval_test"],
                predictions_dir / f"{variant_tag}_code",
                adapter_path=str(adapter_path),
                generate_only=args.generate_only,
            ),
        }
        if not args.skip_finance_test:
            variant_metrics["finance_test"] = evaluate_finance(
                config,
                manifest["finqa_test"],
                predictions_dir / f"{variant_tag}_finance_test.jsonl",
                adapter_path=str(adapter_path),
            )
        conditions[condition_name] = variant_metrics

    baseline = conditions.get("S0")
    baseline_finance_test = baseline["finance_test"]["accuracy"] if baseline and "finance_test" in baseline else None
    baseline_finance_forget_train = baseline["finance_forget_train"]["accuracy"] if baseline else None
    baseline_code = _extract_code_score(baseline["code"]) if baseline else None

    variants_summary: dict[str, object] = {}
    best_finance_test_name = None
    best_finance_test_score = None
    lowest_forget_name = None
    lowest_forget_score = None
    best_code_name = None
    best_code_score = None

    for condition_name, condition_metrics in conditions.items():
        if not condition_name.startswith("S_unlearn"):
            continue
        finance_test_accuracy = (
            condition_metrics["finance_test"]["accuracy"] if "finance_test" in condition_metrics else None
        )
        finance_forget_accuracy = condition_metrics["finance_forget_train"]["accuracy"]
        forget_program_match_rate = condition_metrics["finance_forget_train"].get("forget_program_match_rate")
        forget_term_recall = condition_metrics["finance_forget_train"].get("forget_term_recall")
        forget_refusal_rate = condition_metrics["finance_forget_train"].get("forget_refusal_rate")
        code_pass = _extract_code_score(condition_metrics["code"])
        variant_summary = {
            "finance_test_accuracy": finance_test_accuracy,
            "finance_forget_train_accuracy": finance_forget_accuracy,
            "forget_program_match_rate": forget_program_match_rate,
            "forget_term_recall": forget_term_recall,
            "forget_refusal_rate": forget_refusal_rate,
            "code_pass@1": code_pass,
        }
        if baseline_finance_test is not None and finance_test_accuracy is not None:
            variant_summary["finance_test_delta_vs_base"] = finance_test_accuracy - baseline_finance_test
        if baseline_finance_forget_train is not None:
            variant_summary["finance_forget_train_delta_vs_base"] = (
                finance_forget_accuracy - baseline_finance_forget_train
            )
        if baseline_code is not None and code_pass is not None:
            variant_summary["code_delta_vs_base"] = code_pass - baseline_code
        variants_summary[condition_name] = variant_summary

        if finance_test_accuracy is not None and (
            best_finance_test_score is None or finance_test_accuracy > best_finance_test_score
        ):
            best_finance_test_name = condition_name
            best_finance_test_score = finance_test_accuracy
        if lowest_forget_score is None or finance_forget_accuracy < lowest_forget_score:
            lowest_forget_name = condition_name
            lowest_forget_score = finance_forget_accuracy
        if code_pass is not None and (best_code_score is None or code_pass > best_code_score):
            best_code_name = condition_name
            best_code_score = code_pass

    summary = {
        "student_preset": config.get("resolved_student_preset"),
        "student_model_name_or_path": config["student_model_name_or_path"],
        "baseline_included": args.include_base,
        "finance_test_enabled": not args.skip_finance_test,
        "forget_sample_size": args.forget_sample_size,
        "forget_sample_seed": seed,
        "forget_eval_path": str(forget_eval_path),
        "variants": variants_summary,
        "best_finance_test_variant": best_finance_test_name,
        "best_finance_test_accuracy": best_finance_test_score,
        "lowest_forget_train_variant": lowest_forget_name,
        "lowest_forget_train_accuracy": lowest_forget_score,
        "best_code_variant": best_code_name,
        "best_code_pass@1": best_code_score,
        "conditions": conditions,
    }
    write_json(Path(run_dir) / args.metrics_name, summary)

    lines = [
        "# Unlearning Evaluation Summary",
        f"- StudentPreset: {config.get('resolved_student_preset')}",
        f"- StudentModel: {config['student_model_name_or_path']}",
        f"- BaselineIncluded: {args.include_base}",
        f"- FinanceTestEnabled: {not args.skip_finance_test}",
        f"- ForgetSampleSize: {args.forget_sample_size}",
        f"- ForgetSampleSeed: {seed}",
    ]
    if baseline_finance_test is not None:
        lines.append(f"- BaseFinanceTestAccuracy: {baseline_finance_test}")
    if baseline_finance_forget_train is not None:
        lines.append(f"- BaseFinanceForgetTrainAccuracy: {baseline_finance_forget_train}")
    if baseline_code is not None:
        lines.append(f"- BaseCodePass@1: {baseline_code}")
    for condition_name, variant_summary in variants_summary.items():
        lines.append(
            f"- {condition_name}: finance_test={variant_summary['finance_test_accuracy']}, "
            f"finance_forget_train={variant_summary['finance_forget_train_accuracy']}, "
            f"program_match={variant_summary['forget_program_match_rate']}, "
            f"term_recall={variant_summary['forget_term_recall']}, "
            f"refusal_rate={variant_summary['forget_refusal_rate']}, "
            f"code={variant_summary['code_pass@1']}"
        )
        if "finance_test_delta_vs_base" in variant_summary:
            lines.append(f"- {condition_name}_FinanceTestDeltaVsBase: {variant_summary['finance_test_delta_vs_base']}")
        if "finance_forget_train_delta_vs_base" in variant_summary:
            lines.append(
                f"- {condition_name}_FinanceForgetTrainDeltaVsBase: "
                f"{variant_summary['finance_forget_train_delta_vs_base']}"
            )
        if "code_delta_vs_base" in variant_summary:
            lines.append(f"- {condition_name}_CodeDeltaVsBase: {variant_summary['code_delta_vs_base']}")
    if best_finance_test_name is not None:
        lines.append(f"- BestFinanceTestVariant: {best_finance_test_name}")
    if lowest_forget_name is not None:
        lines.append(f"- LowestForgetTrainVariant: {lowest_forget_name}")
    if best_code_name is not None:
        lines.append(f"- BestCodeVariant: {best_code_name}")
    write_text(Path(run_dir) / args.summary_name, "\n".join(lines))


if __name__ == "__main__":
    main()
