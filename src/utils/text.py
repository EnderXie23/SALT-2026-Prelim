from __future__ import annotations

import math
import re


_NUMERIC_RE = re.compile(r"(?<![\d.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?(?![\d.])")


def normalize_answer(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text


def _clean_answer_span(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\\\[|\\\]$", "", text).strip()
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    text = text.strip("` ")
    return text


def _searchable_text(text: str) -> str:
    return re.sub(r"[*`$]", "", text)


def extract_final_answer(text: str) -> str:
    text = text.strip()
    searchable = _searchable_text(text)
    explicit_patterns = [
        r"final answer\s*[:\-]\s*([^\n]+)",
        r"final\s*[:\-]\s*([^\n]+)",
        r"answer\s*[:\-]\s*([^\n]+)",
    ]
    for pattern in explicit_patterns:
        matches = re.findall(pattern, searchable, flags=re.IGNORECASE)
        if matches:
            return _clean_answer_span(matches[-1].strip())

    closing_lines = [line.strip() for line in searchable.splitlines() if line.strip() and line.strip() not in {r"\[", r"\]"}]
    for line in reversed(closing_lines[-5:]):
        numeric_matches = _NUMERIC_RE.findall(line)
        if numeric_matches:
            return _clean_answer_span(numeric_matches[-1])

    patterns = [
        r"therefore[^\n]*?([-+]?\d+(?:,\d{3})*(?:\.\d+)?%?)",
        r"(?:approx(?:imately)?|about|equals?)\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?%?)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, searchable, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            return _clean_answer_span(matches[-1].strip())

    numeric_matches = _NUMERIC_RE.findall(searchable)
    if numeric_matches:
        return _clean_answer_span(numeric_matches[-1])

    if not closing_lines:
        return ""
    return _clean_answer_span(closing_lines[-1])


def extract_python_code(text: str) -> str:
    text = text.strip()
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced[0].strip()

    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("import ") or stripped.startswith("from "):
            start = idx
            break
    if start is not None:
        return "\n".join(lines[start:]).strip()
    return text


def parse_numeric_answer(text: str) -> tuple[float, bool] | None:
    text = normalize_answer(text)
    match = _NUMERIC_RE.search(text)
    if not match:
        return None
    token = match.group(0).replace(",", "")
    is_percent = token.endswith("%")
    if is_percent:
        token = token[:-1]
    try:
        return float(token), is_percent
    except ValueError:
        return None


def finance_answers_match(prediction: str, gold: str, abs_tol: float = 0.05, rel_tol: float = 0.02) -> bool:
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(gold)
    if norm_pred == norm_gold:
        return True

    pred_num = parse_numeric_answer(prediction)
    gold_num = parse_numeric_answer(gold)
    if pred_num and gold_num:
        pred_value, pred_percent = pred_num
        gold_value, gold_percent = gold_num
        if pred_percent != gold_percent:
            return False
        return math.isclose(pred_value, gold_value, rel_tol=rel_tol, abs_tol=abs_tol)
    return False
