from __future__ import annotations

from tqdm import tqdm

from src.models.generation import generate_text
from src.models.loader import load_student_model, load_tokenizer
from src.utils.io import read_jsonl, write_json, write_jsonl
from src.utils.text import extract_final_answer, finance_answers_match, normalize_answer


def evaluate_finance(config: dict, split_path: str, output_path: str, adapter_path: str | None = None, prepend_context: str | None = None) -> dict:
    tokenizer = load_tokenizer(config["student_model_name_or_path"], config["max_seq_len"])
    model = load_student_model(config, adapter_path=adapter_path, is_trainable=False)
    model.eval()

    rows = read_jsonl(split_path)
    outputs: list[dict] = []
    correct = 0
    for row in tqdm(rows, desc="finance_eval"):
        prompt = row['prompt']
        if prepend_context:
            prompt = (
                f"{prepend_context}\n\n"
                "The notes above are general guidance only. Do not reuse example numbers from them.\n\n"
                f"{prompt}"
            )
        raw = generate_text(
            model,
            tokenizer,
            prompt,
            config["decode_hparams"],
            system_prompt="You answer finance questions succinctly and finish with `Final answer: <answer>`.",
        )
        pred = extract_final_answer(raw)
        is_correct = finance_answers_match(pred, row["target"])
        correct += int(is_correct)
        outputs.append(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "gold": row["target"],
                "raw_prediction": raw,
                "parsed_prediction": pred,
                "normalized_gold": normalize_answer(row["target"]),
                "normalized_prediction": normalize_answer(pred),
                "correct": is_correct,
            }
        )
    metrics = {"accuracy": correct / max(len(rows), 1), "count": len(rows)}
    write_jsonl(output_path, outputs)
    write_json(str(output_path).replace(".jsonl", "_metrics.json"), metrics)
    return metrics
