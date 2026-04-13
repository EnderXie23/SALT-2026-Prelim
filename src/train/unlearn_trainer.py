from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.loader import load_student_model, load_tokenizer, save_adapter
from src.train.dataset import DataCollator, MixedDataset, load_rows, move_batch_to_device
from src.utils.io import ensure_dir, write_json


def _sample_losses(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    loss = loss.view(shift_labels.size())
    mask = (shift_labels != -100).float()
    per_sample = loss.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return per_sample


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_unlearning(config: dict[str, Any], forget_path: str, retain_path: str, output_dir: str) -> dict[str, Any]:
    ensure_dir(output_dir)
    tokenizer = load_tokenizer(config["student_model_name_or_path"], config["max_seq_len"])
    model = load_student_model(config, is_trainable=True)
    model.train()

    system_prompt = "You are a helpful coding assistant. Answer clearly and concisely."
    retain_dataset = MixedDataset(load_rows(retain_path), [], tokenizer, system_prompt)
    forget_dataset = MixedDataset([], load_rows(forget_path), tokenizer, system_prompt)
    loader_kwargs = {
        "batch_size": config["train_hparams"]["per_device_batch_size"],
        "shuffle": True,
        "collate_fn": DataCollator(tokenizer, config["max_seq_len"]),
    }
    retain_loader = DataLoader(retain_dataset, **loader_kwargs)
    forget_loader = DataLoader(forget_dataset, **loader_kwargs)
    retain_iter = iter(retain_loader)
    forget_iter = iter(forget_loader)

    train_cfg = config["train_hparams"]
    optimizer = AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    lambda_forget = float(train_cfg["lambda_forget"])
    retain_weight = float(train_cfg.get("retain_loss_weight", 1.0))
    forget_weight = float(train_cfg.get("forget_loss_weight", 1.0))
    total_steps = int(train_cfg["max_steps"])
    grad_accum = int(train_cfg["gradient_accumulation_steps"])
    save_every = int(train_cfg["save_every_steps"])

    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=total_steps, desc="unlearning")
    running_loss = 0.0
    running_retain_loss = 0.0
    running_forget_loss = 0.0
    for step in range(1, total_steps + 1):
        retain_batch, retain_iter = _next_batch(retain_iter, retain_loader)
        forget_batch, forget_iter = _next_batch(forget_iter, forget_loader)

        retain_batch = move_batch_to_device(retain_batch, model.device)
        forget_batch = move_batch_to_device(forget_batch, model.device)

        retain_outputs = model(input_ids=retain_batch["input_ids"], attention_mask=retain_batch["attention_mask"])
        retain_loss = _sample_losses(retain_outputs.logits, retain_batch["labels"]).mean()

        forget_outputs = model(input_ids=forget_batch["input_ids"], attention_mask=forget_batch["attention_mask"])
        forget_loss = _sample_losses(forget_outputs.logits, forget_batch["labels"]).mean()

        loss = retain_weight * retain_loss - lambda_forget * forget_weight * forget_loss
        (loss / grad_accum).backward()

        if step % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        pbar.update(1)
        loss_value = float(loss.detach().cpu().item())
        retain_loss_value = float(retain_loss.detach().cpu().item())
        forget_loss_value = float(forget_loss.detach().cpu().item())
        running_loss += (loss_value - running_loss) / step
        running_retain_loss += (retain_loss_value - running_retain_loss) / step
        running_forget_loss += (forget_loss_value - running_forget_loss) / step
        pbar.set_postfix(
            loss=f"{loss_value:.4f}",
            retain=f"{retain_loss_value:.4f}",
            forget=f"{forget_loss_value:.4f}",
            avg=f"{running_loss:.4f}",
        )
        history.append(
            {
                "step": step,
                "loss": loss_value,
                "retain_loss": retain_loss_value,
                "forget_loss": forget_loss_value,
                "running_loss": running_loss,
                "running_retain_loss": running_retain_loss,
                "running_forget_loss": running_forget_loss,
            }
        )
        if step % save_every == 0 or step == total_steps:
            ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
            save_adapter(model, ckpt_dir)
            write_json(ckpt_dir / "trainer_state.json", {"history_tail": history[-20:]})
    pbar.close()
    final_dir = Path(output_dir) / "final_adapter"
    save_adapter(model, final_dir)
    metrics = {
        "steps_completed": step,
        "final_loss": history[-1]["loss"] if history else math.nan,
        "final_adapter_path": str(final_dir),
    }
    write_json(Path(output_dir) / "train_metrics.json", metrics)
    write_json(Path(output_dir) / "train_history.json", history)
    return metrics
