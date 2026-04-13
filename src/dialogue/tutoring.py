from __future__ import annotations

import asyncio
import warnings

from openai import AsyncOpenAI
from tqdm import tqdm

from src.models.generation import generate_text
from src.models.loader import load_student_model, load_tokenizer
from src.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl, write_text
from src.utils.text import extract_final_answer, finance_answers_match, normalize_answer


TEACHER_SYSTEM_PROMPT = """You are a QA-only finance tutor.
Explain briefly, ask focused questions, correct mistakes, and help the student recover domain knowledge.
Do not mention or reveal any held-out test items.
Limit your reasoning to 2048 tokens, and your answer to 512 tokens.
If the student answer is already correct, encourage the student to give the answer as the final answer."""


async def _teacher_reply(client: AsyncOpenAI, model: str, messages: list[dict[str, str]], decode_cfg: dict) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=decode_cfg.get("temperature", 0.0),
        top_p=decode_cfg.get("top_p", 1.0),
        max_tokens=decode_cfg.get("teacher_max_new_tokens", 4096),
        extra_body={"repetition_penalty": 1.1},
    )
    content = response.choices[0].message.content
    if content is None:
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        raise ValueError(f"Teacher returned empty content. finish_reason={finish_reason}")
    return content.strip()


def run_tutoring(config: dict, student_checkpoint: str, tutoring_path: str, output_dir: str, teacher_base_url: str, teacher_model: str) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    tokenizer = load_tokenizer(config["student_model_name_or_path"], config["max_seq_len"])
    student = load_student_model(config, adapter_path=student_checkpoint, is_trainable=False)
    student.eval()
    client = AsyncOpenAI(base_url=teacher_base_url, api_key="EMPTY")

    rows = read_jsonl(tutoring_path)[: config["tutoring"]["max_questions"]]
    context_header = (
        "Finance tutoring notes:\n"
        "- Solve only from the current question, not earlier examples.\n"
        "- Extract the requested numerator and denominator before computing.\n"
        "- Preserve units and percentage formatting.\n"
        "- End with: Final answer: <answer>"
    )
    max_concurrent = int(config["tutoring"].get("max_concurrent_sessions", 4))

    async def _run_all() -> list[dict | None]:
        semaphore = asyncio.Semaphore(max_concurrent)
        student_lock = asyncio.Lock()

        async def _generate_student(student_input: str) -> str:
            async with student_lock:
                return await asyncio.to_thread(
                    generate_text,
                    student,
                    tokenizer,
                    student_input,
                    config["decode_hparams"],
                    system_prompt="You are a student learning finance.",
                )

        async def _run_single(row: dict) -> dict | None:
            async with semaphore:
                return await _run_single_session(row, client, teacher_model, config["decode_hparams"], _generate_student)

        tasks = [asyncio.create_task(_run_single(row)) for row in rows]
        results: list[dict | None] = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="tutoring"):
            results.append(await task)
        return results

    async def _run_single_session(row: dict, client: AsyncOpenAI, teacher_model: str, decode_cfg: dict, generate_student_async) -> dict | None:
        messages = [{"role": "system", "content": TEACHER_SYSTEM_PROMPT}]
        student_prompt = f"Question: {row['prompt']}\nAnswer the finance question."
        session_turns: list[dict[str, str]] = []
        student_answer = ""
        first_student_answer = ""
        skip_example = False
        max_turns = config["tutoring"]["max_turns_per_question"]
        for turn_idx in range(max_turns):
            is_last_turn = turn_idx == max_turns - 1
            if turn_idx == 0:
                if is_last_turn:
                    teacher_user = (
                        "This is the only tutoring turn. Give one brief hint, then require the student to provide a final answer.\n\n"
                        f"{row['prompt']}"
                    )
                else:
                    teacher_user = f"Teach the student to solve this question without giving unrelated information.\n\n{row['prompt']}"
            else:
                parsed_student_answer = extract_final_answer(student_answer)
                if finance_answers_match(parsed_student_answer, row["target"]):
                    teacher_answer = "correct"
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The student previously answered:\n"
                                f"{student_answer}\n"
                                "The answer is correct. Reply exactly with: correct"
                            ),
                        }
                    )
                    messages.append({"role": "assistant", "content": teacher_answer})
                    session_turns.append({"speaker": "teacher", "text": teacher_answer})
                    break
                teacher_user = (
                    f"The student previously answered:\n{student_answer}\n"
                    + (
                        "This is the last turn. Give one brief final correction or hint, and tell the student to provide a final answer now."
                        if is_last_turn
                        else "Give a brief correction and one next-step hint."
                    )
            )
            if turn_idx == 0 or not finance_answers_match(extract_final_answer(student_answer), row["target"]):
                messages.append({"role": "user", "content": teacher_user})
                try:
                    teacher_answer = await _teacher_reply(client, teacher_model, messages, decode_cfg)
                except Exception as exc:
                    warnings.warn(f"Skipping tutoring example {row['id']} because teacher generation failed: {exc}")
                    return None
                messages.append({"role": "assistant", "content": teacher_answer})
                session_turns.append({"speaker": "teacher", "text": teacher_answer})
            student_instruction = (
                "Now answer the original question.\n"
                "This is your last turn. You must provide a final answer even if uncertain.\n"
                "End with the format: Final answer: <answer>"
                if is_last_turn
                else "Now answer the original question."
            )
            student_input = f"{teacher_answer}\n\n{student_instruction}\n\n{student_prompt}"
            student_answer = await generate_student_async(student_input)
            if turn_idx == 0:
                first_student_answer = student_answer
            session_turns.append({"speaker": "student", "text": student_answer})
        corrected_target = row["target"]
        teacher_notes = "\n".join(turn["text"] for turn in session_turns if turn["speaker"] == "teacher" and turn["text"].strip().lower() != "correct").strip()
        initial_student_answer = first_student_answer or student_answer
        initial_student_final = extract_final_answer(initial_student_answer)
        final_student_answer = extract_final_answer(student_answer)
        initial_is_correct = finance_answers_match(initial_student_final, corrected_target)
        student_is_correct = finance_answers_match(final_student_answer, corrected_target)
        ft_rows = [
            {
                "id": f"{row['id']}_recover",
                "dataset": "tutoring",
                "domain": "finance",
                "prompt": (
                    "Solve the finance question carefully. Extract the relevant numbers, compute only what is asked, "
                    "and end with `Final answer: <answer>`.\n\n"
                    f"Question: {row['prompt']}"
                ),
                "target": corrected_target,
                "metadata": {"session_id": row["id"], "variant": "plain"},
            }
        ]
        if teacher_notes:
            ft_rows.append(
                {
                    "id": f"{row['id']}_recover_notes",
                    "dataset": "tutoring",
                    "domain": "finance",
                    "prompt": (
                        "Use the tutor notes as general guidance, but solve using only the current question. "
                        "End with `Final answer: <answer>`.\n\n"
                        f"Tutor notes:\n{teacher_notes}\n\nQuestion: {row['prompt']}"
                    ),
                    "target": corrected_target,
                    "metadata": {"session_id": row["id"], "variant": "notes", "student_was_correct": student_is_correct},
                }
            )
        context_snippet = f"Lesson:\n{teacher_notes}" if student_is_correct and teacher_notes else None
        return {
            "session": {
                "id": row["id"],
                "question": row["prompt"],
                "gold_target": corrected_target,
                "initial_student_answer": initial_student_answer,
                "initial_parsed_answer": initial_student_final,
                "initial_correct": initial_is_correct,
                "final_student_answer": student_answer,
                "final_parsed_answer": final_student_answer,
                "final_correct": student_is_correct,
                "turns": session_turns,
            },
            "eval": {
                "id": row["id"],
                "prompt": row["prompt"],
                "gold": corrected_target,
                "initial_raw_prediction": initial_student_answer,
                "initial_parsed_prediction": initial_student_final,
                "initial_normalized_prediction": normalize_answer(initial_student_final),
                "initial_correct": initial_is_correct,
                "final_raw_prediction": student_answer,
                "final_parsed_prediction": final_student_answer,
                "final_normalized_prediction": normalize_answer(final_student_answer),
                "final_correct": student_is_correct,
                "teacher_notes": teacher_notes,
            },
            "ft_rows": ft_rows,
            "context_snippet": context_snippet,
            "initial_correct": int(initial_is_correct),
            "final_correct": int(student_is_correct),
        }

    results = [result for result in asyncio.run(_run_all()) if result is not None]
    sessions = [result["session"] for result in results]
    eval_rows = [result["eval"] for result in results]
    ft_rows = [item for result in results for item in result["ft_rows"]]
    context_snippets = [result["context_snippet"] for result in results if result["context_snippet"]][: config["tutoring"]["context_examples"]]
    initial_correct_total = sum(result["initial_correct"] for result in results)
    final_correct_total = sum(result["final_correct"] for result in results)

    dialogues_path = output_dir / "tutoring_sessions.jsonl"
    eval_path = output_dir / "teach_ctx_eval.jsonl"
    ft_path = output_dir / "teach_ft_data.jsonl"
    ctx_path = output_dir / "teach_context.txt"
    count = len(eval_rows)
    teach_ctx_metrics = {
        "accuracy": final_correct_total / max(count, 1),
        "count": count,
        "initial_accuracy": initial_correct_total / max(count, 1),
        "final_accuracy": final_correct_total / max(count, 1),
        "gain_over_initial": (final_correct_total - initial_correct_total) / max(count, 1),
    }
    write_jsonl(dialogues_path, sessions)
    write_jsonl(eval_path, eval_rows)
    write_jsonl(ft_path, ft_rows)
    write_text(ctx_path, "\n\n".join([context_header, *context_snippets]))
    write_json(output_dir / "teach_ctx_metrics.json", teach_ctx_metrics)
    write_json(
        output_dir / "tutoring_manifest.json",
        {
            "dialogues": str(dialogues_path),
            "teach_ctx_eval": str(eval_path),
            "teach_ctx_metrics": str(output_dir / "teach_ctx_metrics.json"),
            "teach_ft_data": str(ft_path),
            "teach_ctx": str(ctx_path),
        },
    )
    return {
        "dialogues": str(dialogues_path),
        "teach_ctx_eval": str(eval_path),
        "teach_ctx_metrics": str(output_dir / "teach_ctx_metrics.json"),
        "teach_ft_data": str(ft_path),
        "teach_ctx": str(ctx_path),
    }
