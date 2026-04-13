from __future__ import annotations

from typing import Any

import torch


def build_chat_prompt(tokenizer, prompt: str, target: str | None = None, system_prompt: str | None = None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=target is None)
    rendered = ""
    for msg in messages:
        rendered += f"{msg['role'].upper()}: {msg['content']}\n"
    if target is None:
        rendered += "ASSISTANT: "
    return rendered


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, decode_cfg: dict[str, Any], system_prompt: str | None = None) -> str:
    text = build_chat_prompt(tokenizer, prompt=prompt, target=None, system_prompt=system_prompt)
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    batch = {k: v.to(model.device) for k, v in batch.items()}
    outputs = model.generate(
        **batch,
        max_new_tokens=decode_cfg.get("max_new_tokens", 256),
        temperature=decode_cfg.get("temperature", 0.0),
        top_p=decode_cfg.get("top_p", 1.0),
        do_sample=decode_cfg.get("do_sample", False),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][batch["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()