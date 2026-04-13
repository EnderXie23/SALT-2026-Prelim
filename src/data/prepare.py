from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

from src.data.download import download_dataset_from_config
from src.utils.finance_unlearning import build_forget_supervision
from src.utils.io import read_json, read_jsonl, write_json, write_jsonl


def _read_split(spec: str | Path | dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(spec, dict):
        fmt = spec.get("format")
        if fmt in {"json", "jsonl"}:
            return _read_split(spec["path"])
        if fmt == "parquet_dir":
            data_files = spec.get("data_files")
            if not data_files:
                root = Path(spec["path"])
                split_name = spec.get("split")
                if split_name is None:
                    raise ValueError(f"Parquet spec missing split name: {spec}")
                data_files = [str(p) for p in sorted(root.glob(f"{split_name}-*.parquet"))]
            split_name = spec.get("split") or Path(data_files[0]).name.split("-")[0]
            ds = load_dataset("parquet", data_files={split_name: data_files})
            return [dict(row) for row in ds[split_name]]
        raise ValueError(f"Unsupported split spec format: {spec}")

    path = Path(spec)
    if path.suffix == ".json":
        data = read_json(path)
        if isinstance(data, dict):
            for key in ("data", "examples", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        if isinstance(data, list):
            return data
        raise ValueError(f"Unsupported JSON format: {path}")
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    raise ValueError(f"Unsupported split path: {path}")


def _pick(record: dict[str, Any], candidates: Iterable[str], default: str = "") -> Any:
    for key in candidates:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def _render_table(table: Any) -> str:
    if not table:
        return ""
    if isinstance(table, list):
        return "\n".join(" | ".join(str(cell) for cell in row) if isinstance(row, list) else str(row) for row in table)
    return str(table)


def _safe_slice(rows: list[dict[str, Any]], start: int, size: int) -> list[dict[str, Any]]:
    if start >= len(rows):
        return []
    return rows[start : start + size]


def _require_split(manifest: dict[str, Any], dataset_name: str, candidates: list[str]) -> dict[str, Any]:
    splits = manifest.get("splits", {})
    for key in candidates:
        if key in splits:
            spec = splits[key]
            if isinstance(spec, dict) and spec.get("format") and spec.get("path") or spec.get("data_files"):
                return spec
    raise KeyError(f"Dataset '{dataset_name}' is missing required split. Tried {candidates}. Available: {sorted(splits.keys())}")


def _load_or_refresh_manifest(raw_dir: Path, dataset_name: str, config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = raw_dir / dataset_name / "manifest.json"
    source_cfg = config["dataset_sources"][dataset_name]
    needs_refresh = True
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        splits = manifest.get("splits", {})
        if splits:
            needs_refresh = False
    if needs_refresh:
        manifest = download_dataset_from_config(dataset_name, source_cfg, raw_dir, force=True)
    else:
        manifest = read_json(manifest_path)

    expected = {
        "finqa": [["train"], ["test", "validation", "train"]],
        "mbpp": [["train"]],
        "humaneval": [["test"]],
    }[dataset_name]
    for candidates in expected:
        try:
            _require_split(manifest, dataset_name, candidates)
        except KeyError:
            manifest = download_dataset_from_config(dataset_name, source_cfg, raw_dir, force=True)
            _require_split(manifest, dataset_name, candidates)
    return manifest


def normalize_finqa(records: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    items = []
    refusal_template = "I cannot determine the required finance-specific calculation method from the available information."
    for i, record in enumerate(records):
        qa = record.get("qa") if isinstance(record.get("qa"), dict) else {}
        question = _pick(qa, ["question"]) or _pick(record, ["question", "query", "problem", "prompt"])
        answer = _pick(qa, ["answer", "exe_ans"]) or _pick(record, ["answer", "exe_ans", "gold_answer", "label", "target"])
        pre_text = record.get("pre_text", [])
        post_text = record.get("post_text", [])
        if isinstance(pre_text, list):
            pre_text = "\n".join(str(x) for x in pre_text)
        if isinstance(post_text, list):
            post_text = "\n".join(str(x) for x in post_text)
        table_text = _render_table(record.get("table"))
        context_parts = [part for part in [pre_text, table_text, post_text] if part]
        prompt = "\n\n".join(context_parts + [f"Question: {question}"])
        items.append(
            {
                "id": record.get("id", f"finqa_{split_name}_{i}"),
                "dataset": "finqa",
                "domain": "finance",
                "prompt": prompt,
                "target": str(answer),
                "forget_supervision": build_forget_supervision(record, refusal_template=refusal_template),
                "metadata": record,
            }
        )
    return items


def normalize_mbpp(records: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    items = []
    for i, record in enumerate(records):
        prompt = _pick(record, ["text", "prompt", "question"])
        answer = _pick(record, ["code", "canonical_solution", "target"])
        items.append(
            {
                "id": str(record.get("task_id", f"mbpp_{split_name}_{i}")),
                "dataset": "mbpp",
                "domain": "coding",
                "prompt": prompt,
                "target": answer,
                "metadata": record,
            }
        )
    return items


def normalize_humaneval(records: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    items = []
    for i, record in enumerate(records):
        prompt = _pick(record, ["prompt"])
        answer = _pick(record, ["canonical_solution", "solution"])
        items.append(
            {
                "id": str(record.get("task_id", f"humaneval_{split_name}_{i}")),
                "dataset": "humaneval",
                "domain": "coding",
                "prompt": prompt,
                "target": answer,
                "metadata": record,
            }
        )
    return items


def prepare_splits(raw_dir: str | Path, split_dir: str | Path, config: dict[str, Any], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    raw_dir = Path(raw_dir)
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    manifests = {
        "finqa": _load_or_refresh_manifest(raw_dir, "finqa", config),
        "mbpp": _load_or_refresh_manifest(raw_dir, "mbpp", config),
        "humaneval": _load_or_refresh_manifest(raw_dir, "humaneval", config),
    }

    finqa_train_spec = _require_split(manifests["finqa"], "finqa", ["train"])
    finqa_test_spec = _require_split(manifests["finqa"], "finqa", ["test", "validation", "train"])
    mbpp_train_spec = _require_split(manifests["mbpp"], "mbpp", ["train"])
    humaneval_test_spec = _require_split(manifests["humaneval"], "humaneval", ["test"])

    finqa_train = normalize_finqa(_read_split(finqa_train_spec), "train")
    finqa_test = normalize_finqa(_read_split(finqa_test_spec), "test")
    mbpp_train = normalize_mbpp(_read_split(mbpp_train_spec), "train")
    humaneval_test = normalize_humaneval(_read_split(humaneval_test_spec), "test")

    rng.shuffle(finqa_train)
    rng.shuffle(finqa_test)
    rng.shuffle(mbpp_train)
    rng.shuffle(humaneval_test)

    sizes = config["split_sizes"]
    finqa_forget_train = _safe_slice(finqa_train, 0, sizes["finqa_forget_train"])
    finqa_tutor_train = _safe_slice(finqa_train, len(finqa_forget_train), sizes["finqa_tutor_train"])
    outputs = {
        "finqa_forget_train": finqa_forget_train,
        "finqa_tutor_train": finqa_tutor_train,
        "finqa_test": _safe_slice(finqa_test, 0, sizes["finqa_test"]),
        "mbpp_retain_train": _safe_slice(mbpp_train, 0, sizes["mbpp_retain_train"]),
        "humaneval_test": _safe_slice(humaneval_test, 0, sizes["humaneval_test"]),
    }

    index: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    for name, rows in outputs.items():
        path = split_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        index[name] = str(path)
        counts[name] = {
            "requested": sizes.get(name, len(rows)),
            "actual": len(rows),
        }
    write_json(split_dir / "splits_manifest.json", index)
    write_json(split_dir / "splits_summary.json", counts)
    return index
