from __future__ import annotations

import re
from typing import Any


_PROGRAM_OP_RE = re.compile(r"([a-zA-Z_]+)\s*\(")
_WORD_RE = re.compile(r"[a-z][a-z\-]{2,}")
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")
_REFUSAL_PATTERNS = [
    "cannot determine",
    "can't determine",
    "cannot infer",
    "can't infer",
    "insufficient information",
    "not enough information",
    "cannot compute",
    "unable to determine",
    "do not know the finance-specific calculation method",
]
_STOPWORDS = {
    "about",
    "above",
    "after",
    "against",
    "annual",
    "answer",
    "around",
    "asset",
    "assets",
    "based",
    "before",
    "below",
    "between",
    "because",
    "being",
    "cash",
    "clear",
    "clearly",
    "company",
    "concisely",
    "contained",
    "could",
    "course",
    "current",
    "dollar",
    "dollars",
    "during",
    "effect",
    "expense",
    "financial",
    "following",
    "from",
    "general",
    "generally",
    "given",
    "have",
    "helpful",
    "highest",
    "how",
    "item",
    "largest",
    "liability",
    "limited",
    "losses",
    "market",
    "matching",
    "million",
    "notes",
    "number",
    "numbers",
    "obligations",
    "october",
    "only",
    "operations",
    "periods",
    "proportion",
    "question",
    "range",
    "rates",
    "relative",
    "report",
    "result",
    "risk",
    "sales",
    "same",
    "significant",
    "solve",
    "such",
    "table",
    "terms",
    "that",
    "their",
    "these",
    "they",
    "this",
    "through",
    "under",
    "underlying",
    "using",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "year",
}
_FINANCE_KEYWORDS = {
    "amortization",
    "basis",
    "bond",
    "capex",
    "contract",
    "cost",
    "credit",
    "currency",
    "debt",
    "deferred",
    "depreciation",
    "derivative",
    "diluted",
    "discount",
    "dividend",
    "earnings",
    "ebit",
    "ebitda",
    "equity",
    "exchange",
    "expense",
    "exposure",
    "fair",
    "foreign",
    "forward",
    "gross",
    "hedge",
    "income",
    "interest",
    "inventory",
    "investment",
    "libor",
    "liability",
    "margin",
    "maturity",
    "net",
    "notional",
    "operating",
    "payable",
    "principal",
    "ratio",
    "receivable",
    "revenue",
    "settlement",
    "share",
    "stock",
    "tax",
    "variable",
    "yield",
}


def _normalize_text(text: str) -> str:
    return _NON_WORD_RE.sub(" ", str(text).lower()).strip()


def _iter_text_chunks(metadata: dict[str, Any]) -> list[str]:
    chunks: list[str] = []
    qa = metadata.get("qa") if isinstance(metadata.get("qa"), dict) else {}
    question = qa.get("question")
    if question:
        chunks.append(str(question))
    for key in ("pre_text", "post_text"):
        value = metadata.get(key)
        if isinstance(value, list):
            chunks.extend(str(item) for item in value if item)
        elif value:
            chunks.append(str(value))
    table = metadata.get("table")
    if isinstance(table, list):
        for row in table:
            if isinstance(row, list):
                chunks.extend(str(cell) for cell in row if cell)
            elif row:
                chunks.append(str(row))
    return chunks


def parse_program_metadata(program: str) -> dict[str, Any]:
    program = str(program or "").strip()
    operators = [op.lower() for op in _PROGRAM_OP_RE.findall(program)]
    operator_chain = " -> ".join(operators)
    return {
        "program": program,
        "operators": operators,
        "operator_chain": operator_chain,
        "step_count": len(operators),
    }


def extract_finance_terms(metadata: dict[str, Any], max_terms: int = 12) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for chunk in _iter_text_chunks(metadata):
        for token in _WORD_RE.findall(str(chunk).lower()):
            if token in seen or token in _STOPWORDS:
                continue
            if token in _FINANCE_KEYWORDS or token.endswith(("rate", "cost", "gain", "loss", "debt", "cash", "tax")):
                seen.add(token)
                ordered.append(token)
    if not ordered:
        for chunk in _iter_text_chunks(metadata):
            for token in _WORD_RE.findall(str(chunk).lower()):
                if token in seen or token in _STOPWORDS:
                    continue
                seen.add(token)
                ordered.append(token)
                if len(ordered) >= max_terms:
                    return ordered
    return ordered[:max_terms]


def build_forget_supervision(metadata: dict[str, Any], refusal_template: str) -> dict[str, Any]:
    qa = metadata.get("qa") if isinstance(metadata.get("qa"), dict) else {}
    parsed = parse_program_metadata(qa.get("program", ""))
    finance_terms = extract_finance_terms(metadata)
    operator_chain = parsed["operator_chain"] or "unknown"
    program_target = (
        f"Finance-specific method chain: {operator_chain}. "
        f"Step count: {parsed['step_count']}. "
        f"Program: {parsed['program'] or 'unknown'}"
    )
    terms_target = (
        "Finance-specific terms: " + ", ".join(finance_terms)
        if finance_terms
        else "Finance-specific terms: unknown"
    )
    return {
        "program": parsed["program"],
        "program_operators": parsed["operators"],
        "program_operator_chain": operator_chain,
        "program_step_count": parsed["step_count"],
        "finance_terms": finance_terms,
        "program_target": program_target,
        "terms_target": terms_target,
        "refusal_target": refusal_template,
    }


def build_program_forget_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "Explain the finance-specific calculation method needed to solve this problem. "
        "Include the operator sequence and calculation procedure."
    )


def build_terms_forget_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "List the finance-specific terms, concepts, and table cues needed to solve this problem."
    )


def matches_refusal(text: str, refusal_template: str | None = None) -> bool:
    normalized = _normalize_text(text)
    if refusal_template and _normalize_text(refusal_template) in normalized:
        return True
    return any(pattern in normalized for pattern in _REFUSAL_PATTERNS)


def compute_program_match(text: str, metadata: dict[str, Any]) -> bool:
    normalized = _normalize_text(text)
    qa = metadata.get("qa") if isinstance(metadata.get("qa"), dict) else {}
    parsed = parse_program_metadata(qa.get("program", ""))
    if not normalized or not parsed["operators"]:
        return False
    raw_program = _normalize_text(parsed["program"])
    if raw_program and raw_program in normalized:
        return True
    operator_positions: list[int] = []
    cursor = 0
    for operator in parsed["operators"]:
        idx = normalized.find(operator, cursor)
        if idx == -1:
            return False
        operator_positions.append(idx)
        cursor = idx + len(operator)
    return len(operator_positions) == len(parsed["operators"])


def compute_term_recall(text: str, metadata: dict[str, Any]) -> float:
    terms = extract_finance_terms(metadata)
    if not terms:
        return 0.0
    normalized = f" {_normalize_text(text)} "
    hits = sum(1 for term in terms if f" {term} " in normalized)
    return hits / len(terms)


def summarize_forget_predictions(
    rows_by_id: dict[str, dict[str, Any]],
    predictions: list[dict[str, Any]],
    refusal_template: str | None = None,
) -> dict[str, Any]:
    total = 0
    program_matches = 0
    refusal_matches = 0
    term_recall_sum = 0.0
    for prediction in predictions:
        row = rows_by_id.get(str(prediction.get("id")))
        if not row:
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        raw_prediction = str(prediction.get("raw_prediction", ""))
        total += 1
        program_matches += int(compute_program_match(raw_prediction, metadata))
        refusal_matches += int(matches_refusal(raw_prediction, refusal_template=refusal_template))
        term_recall_sum += compute_term_recall(raw_prediction, metadata)
    if total == 0:
        return {
            "forget_program_match_rate": 0.0,
            "forget_term_recall": 0.0,
            "forget_refusal_rate": 0.0,
        }
    return {
        "forget_program_match_rate": program_matches / total,
        "forget_term_recall": term_recall_sum / total,
        "forget_refusal_rate": refusal_matches / total,
    }
