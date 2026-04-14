from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.generation import build_chat_prompt
from src.models.loader import load_student_model, load_tokenizer, save_adapter
from src.train.dataset import DataCollator, MixedDataset, load_rows, move_batch_to_device
from src.utils.finance_unlearning import build_forget_supervision, build_program_forget_prompt, build_terms_forget_prompt
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


def _estimate_loader_loss(model, loader, max_batches: int) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            batch = move_batch_to_device(batch, model.device)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            losses.append(float(_sample_losses(outputs.logits, batch["labels"]).mean().detach().cpu().item()))
    if was_training:
        model.train()
    if not losses:
        return math.nan
    return sum(losses) / len(losses)


def _masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
    return (loss.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).mean()


def _encode_supervision_batch(
    tokenizer,
    prompts: list[str],
    targets: list[str],
    system_prompt: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    texts = [
        build_chat_prompt(tokenizer, prompt=prompt, target=target, system_prompt=system_prompt)
        for prompt, target in zip(prompts, targets, strict=False)
    ]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    encoded["labels"] = labels
    return encoded


def _aux_target_loss(
    model,
    tokenizer,
    prompts: list[str],
    targets: list[str],
    system_prompt: str,
    max_length: int,
) -> torch.Tensor:
    batch = _encode_supervision_batch(tokenizer, prompts, targets, system_prompt=system_prompt, max_length=max_length)
    batch = {key: value.to(model.device) for key, value in batch.items()}
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return _masked_ce_loss(outputs.logits, batch["labels"])


def _estimate_aux_loss(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    prompt_builder,
    target_key: str,
    system_prompt: str,
    max_length: int,
    max_batches: int,
    batch_size: int,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, min(len(rows), max_batches * batch_size), batch_size):
            chunk = rows[start : start + batch_size]
            prompts = [prompt_builder(row["prompt"]) for row in chunk]
            targets = [row["forget_supervision"][target_key] for row in chunk]
            loss = _aux_target_loss(model, tokenizer, prompts, targets, system_prompt=system_prompt, max_length=max_length)
            losses.append(float(loss.detach().cpu().item()))
    if was_training:
        model.train()
    if not losses:
        return math.nan
    return sum(losses) / len(losses)


def _build_forget_objective(train_cfg: dict[str, Any], model, forget_loader) -> tuple[str, dict[str, float]]:
    objective = str(train_cfg.get("forget_objective", "gradient_ascent")).lower()
    objective_meta: dict[str, float] = {}
    if objective in {"gradient_ascent", "method_term_aux"}:
        return objective, objective_meta
    if objective != "threshold_hinge":
        raise ValueError(f"Unsupported forget_objective: {objective}")

    estimate_batches = int(train_cfg.get("forget_loss_estimate_batches", 32))
    margin = float(train_cfg.get("forget_loss_margin", 1.0))
    baseline_forget_loss = _estimate_loader_loss(model, forget_loader, max_batches=estimate_batches)
    target_override = train_cfg.get("forget_loss_target")
    target_forget_loss = float(target_override) if target_override is not None else baseline_forget_loss + margin
    objective_meta = {
        "baseline_forget_loss": baseline_forget_loss,
        "forget_loss_margin": margin,
        "target_forget_loss": target_forget_loss,
        "forget_loss_estimate_batches": float(estimate_batches),
    }
    return objective, objective_meta


def _build_method_term_meta(
    train_cfg: dict[str, Any],
    model,
    tokenizer,
    forget_rows: list[dict[str, Any]],
    max_length: int,
    batch_size: int,
) -> dict[str, float | str]:
    penalty_mode = str(train_cfg.get("method_term_penalty_mode", "gradient_ascent")).lower()
    meta: dict[str, float | str] = {"method_term_penalty_mode": penalty_mode}
    if penalty_mode == "gradient_ascent":
        return meta
    if penalty_mode != "threshold_hinge":
        raise ValueError(f"Unsupported method_term_penalty_mode: {penalty_mode}")
    estimate_batches = int(train_cfg.get("forget_loss_estimate_batches", 32))
    program_margin = float(train_cfg.get("program_loss_margin", 1.0))
    term_margin = float(train_cfg.get("term_loss_margin", 0.5))
    refusal_margin = float(train_cfg.get("refusal_loss_margin", 0.0))
    system_prompt = "You are a helpful coding assistant. Answer clearly and concisely."
    baseline_program_loss = _estimate_aux_loss(
        model,
        tokenizer,
        forget_rows,
        build_program_forget_prompt,
        "program_target",
        system_prompt=system_prompt,
        max_length=max_length,
        max_batches=estimate_batches,
        batch_size=batch_size,
    )
    baseline_term_loss = _estimate_aux_loss(
        model,
        tokenizer,
        forget_rows,
        build_terms_forget_prompt,
        "terms_target",
        system_prompt=system_prompt,
        max_length=max_length,
        max_batches=estimate_batches,
        batch_size=batch_size,
    )
    refusal_template = str(train_cfg.get("forget_refusal_template", "")).strip()
    baseline_refusal_loss = _estimate_aux_loss(
        model,
        tokenizer,
        forget_rows,
        lambda prompt: prompt,
        "refusal_target",
        system_prompt=system_prompt,
        max_length=max_length,
        max_batches=estimate_batches,
        batch_size=batch_size,
    )
    meta.update(
        {
            "forget_loss_estimate_batches": float(estimate_batches),
            "baseline_program_loss": baseline_program_loss,
            "baseline_term_loss": baseline_term_loss,
            "baseline_refusal_loss": baseline_refusal_loss,
            "target_program_loss": baseline_program_loss + program_margin,
            "target_term_loss": baseline_term_loss + term_margin,
            "target_refusal_loss": max(0.0, baseline_refusal_loss - refusal_margin),
            "refusal_template": refusal_template,
        }
    )
    return meta


def _text_penalty(loss: torch.Tensor, penalty_mode: str, target_value: float | None = None) -> torch.Tensor:
    if penalty_mode == "gradient_ascent":
        return -loss
    if penalty_mode == "threshold_hinge":
        if target_value is None:
            raise RuntimeError("threshold_hinge penalty requires target_value.")
        target = torch.tensor(target_value, device=loss.device, dtype=loss.dtype)
        return F.relu(target - loss)
    raise RuntimeError(f"Unexpected penalty mode: {penalty_mode}")


def train_unlearning(config: dict[str, Any], forget_path: str, retain_path: str, output_dir: str) -> dict[str, Any]:
    ensure_dir(output_dir)
    tokenizer = load_tokenizer(config["student_model_name_or_path"], config["max_seq_len"])
    model = load_student_model(config, is_trainable=True)
    model.train()

    system_prompt = "You are a helpful coding assistant. Answer clearly and concisely."
    retain_rows = load_rows(retain_path)
    forget_rows = load_rows(forget_path)
    refusal_template = str(
        config["train_hparams"].get(
            "forget_refusal_template",
            "I cannot determine the required finance-specific calculation method from the available information.",
        )
    )
    for row in forget_rows:
        metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        if not row.get("forget_supervision"):
            row["forget_supervision"] = build_forget_supervision(metadata, refusal_template=refusal_template)
        else:
            row["forget_supervision"].setdefault("refusal_target", refusal_template)

    retain_dataset = MixedDataset(retain_rows, [], tokenizer, system_prompt)
    forget_dataset = MixedDataset([], forget_rows, tokenizer, system_prompt)
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
    forget_objective, forget_objective_meta = _build_forget_objective(train_cfg, model, forget_loader)
    target_forget_loss = forget_objective_meta.get("target_forget_loss")
    lambda_program = float(train_cfg.get("lambda_program", 1.5))
    lambda_terms = float(train_cfg.get("lambda_terms", 0.75))
    lambda_refusal = float(train_cfg.get("lambda_refusal", 0.5))
    method_term_meta: dict[str, float | str] = {}
    if forget_objective == "method_term_aux":
        method_term_meta = _build_method_term_meta(
            train_cfg,
            model,
            tokenizer,
            forget_rows,
            max_length=config["max_seq_len"],
            batch_size=config["train_hparams"]["per_device_batch_size"],
        )

    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=total_steps, desc="unlearning")
    running_loss = 0.0
    running_retain_loss = 0.0
    running_forget_loss = 0.0
    running_forget_penalty = 0.0
    running_program_loss = 0.0
    running_term_loss = 0.0
    running_refusal_loss = 0.0
    for step in range(1, total_steps + 1):
        retain_batch, retain_iter = _next_batch(retain_iter, retain_loader)
        forget_batch, forget_iter = _next_batch(forget_iter, forget_loader)

        retain_batch = move_batch_to_device(retain_batch, model.device)
        forget_batch = move_batch_to_device(forget_batch, model.device)

        if forget_objective == "method_term_aux":
            retain_outputs = model(input_ids=retain_batch["input_ids"], attention_mask=retain_batch["attention_mask"])
            retain_loss = _sample_losses(retain_outputs.logits, retain_batch["labels"]).mean()
            (retain_weight * retain_loss / grad_accum).backward()
            del retain_outputs

            with torch.no_grad():
                forget_outputs = model(input_ids=forget_batch["input_ids"], attention_mask=forget_batch["attention_mask"])
                forget_loss = _sample_losses(forget_outputs.logits, forget_batch["labels"]).mean()
            del forget_outputs

            program_loss = torch.zeros((), device=model.device)
            term_loss = torch.zeros((), device=model.device)
            refusal_loss = torch.zeros((), device=model.device)
            penalty_mode = str(method_term_meta.get("method_term_penalty_mode", "gradient_ascent"))
            program_prompts = [build_program_forget_prompt(prompt) for prompt in forget_batch["prompts"]]
            program_targets = [item["program_target"] for item in forget_batch["forget_supervision"]]
            program_loss = _aux_target_loss(
                model,
                tokenizer,
                program_prompts,
                program_targets,
                system_prompt=system_prompt,
                max_length=config["max_seq_len"],
            )
            program_penalty = _text_penalty(
                program_loss,
                penalty_mode,
                target_value=float(method_term_meta["target_program_loss"]) if "target_program_loss" in method_term_meta else None,
            )
            (forget_weight * lambda_program * program_penalty / grad_accum).backward()
            program_penalty_value = program_penalty.detach()
            del program_penalty

            term_prompts = [build_terms_forget_prompt(prompt) for prompt in forget_batch["prompts"]]
            term_targets = [item["terms_target"] for item in forget_batch["forget_supervision"]]
            term_loss = _aux_target_loss(
                model,
                tokenizer,
                term_prompts,
                term_targets,
                system_prompt=system_prompt,
                max_length=config["max_seq_len"],
            )
            term_penalty = _text_penalty(
                term_loss,
                penalty_mode,
                target_value=float(method_term_meta["target_term_loss"]) if "target_term_loss" in method_term_meta else None,
            )
            (forget_weight * lambda_terms * term_penalty / grad_accum).backward()
            term_penalty_value = term_penalty.detach()
            del term_penalty

            refusal_targets = [item["refusal_target"] for item in forget_batch["forget_supervision"]]
            refusal_loss = _aux_target_loss(
                model,
                tokenizer,
                forget_batch["prompts"],
                refusal_targets,
                system_prompt=system_prompt,
                max_length=config["max_seq_len"],
            )
            (forget_weight * lambda_refusal * refusal_loss / grad_accum).backward()
            forget_penalty = lambda_program * program_penalty_value + lambda_terms * term_penalty_value + lambda_refusal * refusal_loss.detach()
            loss = retain_weight * retain_loss.detach() + forget_weight * forget_penalty
        else:
            retain_outputs = model(input_ids=retain_batch["input_ids"], attention_mask=retain_batch["attention_mask"])
            retain_loss = _sample_losses(retain_outputs.logits, retain_batch["labels"]).mean()

            forget_outputs = model(input_ids=forget_batch["input_ids"], attention_mask=forget_batch["attention_mask"])
            forget_loss = _sample_losses(forget_outputs.logits, forget_batch["labels"]).mean()

            program_loss = torch.zeros((), device=forget_loss.device)
            term_loss = torch.zeros((), device=forget_loss.device)
            refusal_loss = torch.zeros((), device=forget_loss.device)

            if forget_objective == "gradient_ascent":
                forget_penalty = -forget_loss
            elif forget_objective == "threshold_hinge":
                if target_forget_loss is None:
                    raise RuntimeError("threshold_hinge objective requires a target_forget_loss.")
                forget_penalty = F.relu(
                    torch.tensor(target_forget_loss, device=forget_loss.device, dtype=forget_loss.dtype) - forget_loss
                )
            else:
                raise RuntimeError(f"Unexpected forget objective: {forget_objective}")
            loss = retain_weight * retain_loss + lambda_forget * forget_weight * forget_penalty
            (loss / grad_accum).backward()

        if step % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        pbar.update(1)
        loss_value = float(loss.detach().cpu().item())
        retain_loss_value = float(retain_loss.detach().cpu().item())
        forget_loss_value = float(forget_loss.detach().cpu().item())
        forget_penalty_value = float(forget_penalty.detach().cpu().item())
        program_loss_value = float(program_loss.detach().cpu().item())
        term_loss_value = float(term_loss.detach().cpu().item())
        refusal_loss_value = float(refusal_loss.detach().cpu().item())
        running_loss += (loss_value - running_loss) / step
        running_retain_loss += (retain_loss_value - running_retain_loss) / step
        running_forget_loss += (forget_loss_value - running_forget_loss) / step
        running_forget_penalty += (forget_penalty_value - running_forget_penalty) / step
        running_program_loss += (program_loss_value - running_program_loss) / step
        running_term_loss += (term_loss_value - running_term_loss) / step
        running_refusal_loss += (refusal_loss_value - running_refusal_loss) / step
        pbar.set_postfix(
            loss=f"{loss_value:.4f}",
            retain=f"{retain_loss_value:.4f}",
            forget=f"{forget_loss_value:.4f}",
            forget_pen=f"{forget_penalty_value:.4f}",
            avg=f"{running_loss:.4f}",
        )
        history.append(
            {
                "step": step,
                "loss": loss_value,
                "retain_loss": retain_loss_value,
                "forget_loss": forget_loss_value,
                "forget_penalty": forget_penalty_value,
                "program_loss": program_loss_value,
                "term_loss": term_loss_value,
                "refusal_loss": refusal_loss_value,
                "running_loss": running_loss,
                "running_retain_loss": running_retain_loss,
                "running_forget_loss": running_forget_loss,
                "running_forget_penalty": running_forget_penalty,
                "running_program_loss": running_program_loss,
                "running_term_loss": running_term_loss,
                "running_refusal_loss": running_refusal_loss,
                "forget_objective": forget_objective,
                "target_forget_loss": target_forget_loss,
            }
        )
        if step % save_every == 0 or step == total_steps:
            ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
            save_adapter(model, ckpt_dir)
            write_json(
                ckpt_dir / "trainer_state.json",
                {
                    "history_tail": history[-20:],
                    "forget_objective": forget_objective,
                    "forget_objective_meta": forget_objective_meta,
                    "method_term_meta": method_term_meta,
                },
            )
    pbar.close()
    final_dir = Path(output_dir) / "final_adapter"
    save_adapter(model, final_dir)
    metrics = {
        "steps_completed": step,
        "final_loss": history[-1]["loss"] if history else math.nan,
        "final_adapter_path": str(final_dir),
        "forget_objective": forget_objective,
        "forget_objective_meta": forget_objective_meta,
        "method_term_meta": method_term_meta,
    }
    write_json(Path(output_dir) / "train_metrics.json", metrics)
    write_json(Path(output_dir) / "train_history.json", history)
    return metrics
