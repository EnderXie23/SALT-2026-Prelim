from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping[dtype_name.lower()]


def _require_bitsandbytes() -> None:
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError("Quantization is enabled but bitsandbytes is not installed. Set use_quantization=false or install bitsandbytes.")


def load_tokenizer(model_name_or_path: str, max_seq_len: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_len
    return tokenizer


def load_student_model(config: dict[str, Any], adapter_path: str | None = None, is_trainable: bool = True):
    model_name = config["student_model_name_or_path"]
    torch_dtype = resolve_dtype(config["torch_dtype"])
    use_quant = bool(config.get("use_quantization", True))
    quant_bits = int(config.get("quantization_bits", 4))

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }
    if use_quant:
        _require_bitsandbytes()
        if quant_bits != 4:
            raise ValueError("This scaffold currently supports 4-bit quantization only.")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if use_quant and is_trainable:
        model = prepare_model_for_kbit_training(model)

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable)
    elif is_trainable:
        hparams = config["train_hparams"]
        lora_config = LoraConfig(
            r=hparams["lora_rank"],
            lora_alpha=hparams["lora_alpha"],
            lora_dropout=hparams["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=hparams["target_modules"],
        )
        model = get_peft_model(model, lora_config)

    return model


def save_adapter(model, output_dir: str | Path) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)