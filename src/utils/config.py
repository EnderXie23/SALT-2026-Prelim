from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def apply_student_preset(config: Dict[str, Any], preset_name: str | None) -> Dict[str, Any]:
    presets = config.get("model_presets") or {}
    active = preset_name or config.get("active_student_model")
    if active:
        if active not in presets:
            raise KeyError(f"Unknown student preset: {active}")
        config = _deep_merge(config, presets[active])
        config["resolved_student_preset"] = active
    elif presets and "student_model_name_or_path" not in config:
        raise KeyError("Config defines model_presets but no active_student_model.")
    return config


def load_config(path: str | Path, student_preset: str | None = None) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return apply_student_preset(config, student_preset)