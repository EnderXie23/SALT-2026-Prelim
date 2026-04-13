from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset

from src.utils.io import ensure_dir, read_json, write_json


def _save_dataset_dict(dataset_dict: DatasetDict, out_dir: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {"splits": {}}
    ensure_dir(out_dir)
    for split, ds in dataset_dict.items():
        split_path = out_dir / f"{split}.jsonl"
        ds.to_json(str(split_path), force_ascii=False)
        manifest["splits"][split] = {"path": str(split_path), "format": "jsonl", "count": len(ds)}
    return manifest


def _resolve_finqa_local(local_root: Path) -> dict[str, Any]:
    raw_root = local_root / "raw"
    candidates = {
        "train": raw_root / "train.json",
        "validation": raw_root / "dev.json",
        "test": raw_root / "test.json",
        "private_test": raw_root / "private_test.json",
    }
    splits = {}
    for split_name, path in candidates.items():
        if path.exists():
            count = len(read_json(path))
            splits[split_name] = {"path": str(path), "format": "json", "count": count}
    if not splits:
        raise RuntimeError(f"No FinQA raw JSON files found under {raw_root}")
    return splits


def _resolve_local_parquet_splits(local_root: Path, split_map: dict[str, str] | None = None) -> dict[str, Any]:
    split_map = split_map or {}
    splits = {}
    for logical_split, disk_split in split_map.items():
        files = sorted(local_root.glob(f"{disk_split}-*.parquet"))
        if files:
            ds = load_dataset("parquet", data_files={logical_split: [str(p) for p in files]})
            splits[logical_split] = {
                "path": str(files[0].parent),
                "format": "parquet_dir",
                "data_files": [str(p) for p in files],
                "count": len(ds[logical_split]),
            }
    if not splits:
        direct_files = sorted(local_root.glob("*.parquet"))
        if direct_files:
            grouped: dict[str, list[str]] = {}
            for file in direct_files:
                split_name = file.name.split("-")[0]
                grouped.setdefault(split_name, []).append(str(file))
            ds = load_dataset("parquet", data_files=grouped)
            for split_name, files in grouped.items():
                splits[split_name] = {
                    "path": str(local_root),
                    "format": "parquet_dir",
                    "data_files": files,
                    "count": len(ds[split_name]),
                }
    if not splits:
        raise RuntimeError(f"No parquet split files found under {local_root}")
    return splits


def _build_local_manifest(name: str, local_root: Path, source_cfg: dict[str, Any]) -> dict[str, Any]:
    if name == "finqa":
        splits = _resolve_finqa_local(local_root)
    elif name == "mbpp":
        config_name = source_cfg.get("preferred_config") or "sanitized"
        config_root = local_root / config_name
        if not config_root.exists():
            available = [p.name for p in local_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
            raise RuntimeError(f"MBPP config directory '{config_name}' not found under {local_root}. Available: {available}")
        splits = _resolve_local_parquet_splits(
            config_root,
            split_map=source_cfg.get("split_map") or {"train": "train", "validation": "validation", "test": "test", "prompt": "prompt"},
        )
    elif name == "humaneval":
        config_name = source_cfg.get("preferred_config") or "openai_humaneval"
        config_root = local_root / config_name
        if not config_root.exists():
            available = [p.name for p in local_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
            raise RuntimeError(f"HumanEval config directory '{config_name}' not found under {local_root}. Available: {available}")
        splits = _resolve_local_parquet_splits(
            config_root,
            split_map=source_cfg.get("split_map") or {"test": "test"},
        )
    else:
        raise ValueError(f"Unsupported local dataset manifest builder for {name}")

    return {
        "dataset": name,
        "source_type": "local",
        "source": str(local_root),
        "splits": splits,
    }


def download_dataset_from_config(name: str, source_cfg: dict[str, Any], out_dir: str | Path, force: bool = False) -> dict[str, Any]:
    out_dir = Path(out_dir) / name
    ensure_dir(out_dir)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not force:
        return read_json(manifest_path)

    local_path = source_cfg.get("local_path")
    if not local_path:
        inferred_local = out_dir
        if inferred_local.exists() and any(inferred_local.iterdir()):
            local_path = str(inferred_local)

    if local_path:
        manifest = _build_local_manifest(name, Path(local_path), source_cfg)
        write_json(manifest_path, manifest)
        return manifest

    errors: list[str] = []
    for candidate in source_cfg.get("hf_candidates", []):
        try:
            load_kwargs: dict[str, Any] = {}
            preferred_config = source_cfg.get("preferred_config")
            if preferred_config:
                load_kwargs["name"] = preferred_config
            loaded = load_dataset(candidate, **load_kwargs)
            if not isinstance(loaded, DatasetDict):
                if "train" in loaded:
                    loaded = DatasetDict({"train": loaded["train"]})
                else:
                    raise ValueError(f"Unsupported dataset type for {candidate}")
            manifest = {
                "dataset": name,
                "source_type": "huggingface",
                "source": candidate,
                "preferred_config": preferred_config,
            }
            manifest.update(_save_dataset_dict(loaded, out_dir))
            write_json(manifest_path, manifest)
            return manifest
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError(f"Unable to download or resolve dataset '{name}'. Tried local path and HF candidates: {errors}")