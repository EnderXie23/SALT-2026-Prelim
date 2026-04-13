from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from src.models.generation import build_chat_prompt
from src.utils.io import read_jsonl


@dataclass
class TrainSample:
    sample_id: str
    source: str
    text: str


class SFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, source: str, system_prompt: str):
        self.samples: list[TrainSample] = []
        for row in rows:
            text = build_chat_prompt(tokenizer, prompt=row["prompt"], target=row["target"], system_prompt=system_prompt)
            self.samples.append(TrainSample(sample_id=row["id"], source=source, text=text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TrainSample:
        return self.samples[idx]


class MixedDataset(Dataset):
    def __init__(self, retain_rows: list[dict[str, Any]], forget_rows: list[dict[str, Any]], tokenizer, system_prompt: str):
        self.samples = SFTDataset(retain_rows, tokenizer, "retain", system_prompt).samples + SFTDataset(
            forget_rows, tokenizer, "forget", system_prompt
        ).samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TrainSample:
        return self.samples[idx]


class DataCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[TrainSample]) -> dict[str, Any]:
        texts = [sample.text for sample in batch]
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "sources": [sample.source for sample in batch],
            "sample_ids": [sample.sample_id for sample in batch],
        }


def load_rows(path: str) -> list[dict[str, Any]]:
    return read_jsonl(path)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
        "sources": batch["sources"],
        "sample_ids": batch["sample_ids"],
    }