from __future__ import annotations

from pathlib import Path

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.loader import load_student_model, load_tokenizer, save_adapter
from src.train.dataset import DataCollator, SFTDataset, load_rows, move_batch_to_device
from src.train.unlearn_trainer import _sample_losses
from src.utils.io import ensure_dir, write_json


def train_recovery(config: dict, adapter_path: str, tutoring_data_path: str, output_dir: str) -> dict:
    ensure_dir(output_dir)
    tokenizer = load_tokenizer(config["student_model_name_or_path"], config["max_seq_len"])
    model = load_student_model(config, adapter_path=adapter_path, is_trainable=True)
    model.train()

    rows = load_rows(tutoring_data_path)
    dataset = SFTDataset(rows, tokenizer, "retain", "You are a helpful student learning finance from a tutor.")
    loader = DataLoader(
        dataset,
        batch_size=config["train_hparams"]["per_device_batch_size"],
        shuffle=True,
        collate_fn=DataCollator(tokenizer, config["max_seq_len"]),
    )

    train_cfg = config["train_hparams"]
    optimizer = AdamW(model.parameters(), lr=float(train_cfg["recovery_learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    total_steps = int(train_cfg["recovery_max_steps"])
    grad_accum = int(train_cfg["gradient_accumulation_steps"])
    step = 0
    history: list[dict] = []
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=total_steps, desc="recovery")
    running_loss = 0.0
    while step < total_steps:
        for batch in loader:
            batch = move_batch_to_device(batch, model.device)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = _sample_losses(outputs.logits, batch["labels"]).mean()
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            pbar.update(1)
            loss_value = float(loss.detach().cpu().item())
            running_loss += (loss_value - running_loss) / step
            pbar.set_postfix(loss=f"{loss_value:.4f}", avg=f"{running_loss:.4f}")
            history.append({"step": step, "loss": loss_value, "running_loss": running_loss})
            if step >= total_steps:
                break
        else:
            continue
        break
    pbar.close()
    final_dir = Path(output_dir) / "final_adapter"
    save_adapter(model, final_dir)
    metrics = {"steps_completed": step, "final_adapter_path": str(final_dir)}
    write_json(Path(output_dir) / "train_metrics.json", metrics)
    write_json(Path(output_dir) / "train_history.json", history)
    return metrics
