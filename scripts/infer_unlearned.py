from __future__ import annotations

import argparse

from _bootstrap import setup_project_root

setup_project_root()

from src.models.generation import generate_text
from src.models.loader import load_student_model, load_tokenizer
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--student-preset", default=None)
    args = parser.parse_args()

    if not args.prompt and not args.interactive:
        raise ValueError("Pass --prompt for a one-shot query or use --interactive.")

    config = load_config(args.config, student_preset=args.student_preset)
    tokenizer = load_tokenizer(config["student_model_name_or_path"], config["max_seq_len"])
    model = load_student_model(config, adapter_path=args.adapter_path, is_trainable=False)
    model.eval()

    decode_cfg = dict(config["decode_hparams"])
    if args.max_new_tokens is not None:
        decode_cfg["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        decode_cfg["temperature"] = args.temperature
    if args.top_p is not None:
        decode_cfg["top_p"] = args.top_p
    if args.do_sample:
        decode_cfg["do_sample"] = True

    def run_once(prompt: str) -> None:
        response = generate_text(
            model,
            tokenizer,
            prompt,
            decode_cfg,
            system_prompt=args.system_prompt,
        )
        print("\n=== Response ===")
        print(response)
        print()

    if args.prompt:
        run_once(args.prompt)
        if not args.interactive:
            return

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break
        run_once(prompt)


if __name__ == "__main__":
    main()
