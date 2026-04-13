from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

from src.models.generation import generate_text
from src.models.loader import load_student_model, load_tokenizer
from src.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl
from src.utils.text import extract_python_code


def _render_completion_row(row: dict, completion: str) -> dict:
    task_id = row["metadata"].get("task_id") or row["id"]
    return {"task_id": task_id, "completion": extract_python_code(completion)}


def _render_problem_row(row: dict) -> dict:
    metadata = row["metadata"]
    return {
        "task_id": metadata.get("task_id") or row["id"],
        "prompt": metadata["prompt"],
        "canonical_solution": metadata.get("canonical_solution", row.get("target", "")),
        "test": metadata["test"],
        "entry_point": metadata["entry_point"],
    }


def _parse_humaneval_metrics(stdout: str) -> dict:
    parsed_metrics: dict = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not (line.startswith("{") and "pass@1" in line):
            continue
        try:
            payload = json.loads(line)
            parsed_metrics.update(payload)
            break
        except json.JSONDecodeError:
            pass
        try:
            sanitized = re.sub(r"np\.float64\(([^\)]+)\)", r"\1", line)
            payload = ast.literal_eval(sanitized)
            parsed_metrics.update(payload)
            break
        except (ValueError, SyntaxError):
            pass
        match = re.search(r"pass@1'?:\s*(?:np\.float64\()?([0-9]+(?:\.[0-9]+)?)", line)
        if match:
            parsed_metrics["pass@1"] = float(match.group(1))
            break
    if "pass@1" in parsed_metrics:
        parsed_metrics["pass@1"] = float(parsed_metrics["pass@1"])
    return parsed_metrics


def _attempt_humaneval_exec(completions_path: Path, problems_path: Path, output_dir: Path) -> dict:
    metrics = {
        "execution_mode": "generate_only",
        "pass@1": None,
        "command": [
            sys.executable,
            "-m",
            "human_eval.evaluate_functional_correctness",
            str(completions_path),
            f"--problem_file={problems_path}",
        ],
    }
    cmd = metrics["command"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metrics["execution_mode"] = "human_eval"
        metrics["returncode"] = result.returncode
        metrics["stdout"] = result.stdout
        metrics["stderr"] = result.stderr
        metrics.update(_parse_humaneval_metrics(result.stdout))
    except subprocess.CalledProcessError as exc:
        metrics["returncode"] = exc.returncode
        metrics["stdout"] = exc.stdout
        metrics["stderr"] = exc.stderr
        metrics["error"] = str(exc)
        metrics.update(_parse_humaneval_metrics(exc.stdout or ""))
    except Exception as exc:
        metrics["error"] = str(exc)
    write_json(output_dir / "humaneval_exec_metrics.json", metrics)
    return metrics


def evaluate_code(config: dict, split_path: str, output_dir: str, adapter_path: str | None = None, generate_only: bool = False) -> dict:
    output_dir = ensure_dir(output_dir)
    tokenizer = load_tokenizer(config["student_model_name_or_path"], config["max_seq_len"])
    model = load_student_model(config, adapter_path=adapter_path, is_trainable=False)
    model.eval()

    rows = read_jsonl(split_path)
    predictions = []
    problems = []
    for row in tqdm(rows, desc="code_eval"):
        prompt = f"{row['prompt']}\n\nReturn only Python code."
        completion = generate_text(model, tokenizer, prompt, config["decode_hparams"], system_prompt="You write correct Python functions.")
        predictions.append(_render_completion_row(row, completion))
        problems.append(_render_problem_row(row))
    completions_path = output_dir / "humaneval_predictions.jsonl"
    problems_path = output_dir / "humaneval_problems.jsonl"
    write_jsonl(completions_path, predictions)
    write_jsonl(problems_path, problems)
    if generate_only:
        metrics = {"execution_mode": "generate_only", "count": len(predictions)}
        write_json(output_dir / "humaneval_exec_metrics.json", metrics)
        return metrics
    return _attempt_humaneval_exec(completions_path, problems_path, output_dir)