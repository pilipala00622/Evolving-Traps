from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from core.hard_hallucination_schema import HardHallucinationReviewTask
from hard_hallucination_config import REVIEW_BOOL_FIELDS


DEFAULT_CARDS = "data/hard_hallucination/hard_hallucination_cards.jsonl"
DEFAULT_CONTEXTS = "data/hard_hallucination/source_contexts.jsonl"
DEFAULT_OUTPUT = "data/hard_hallucination/review/review_tasks.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _stable_id(*parts: str, prefix: str) -> str:
    digest = hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}{digest}"


def build_review_tasks(cards_path: Path | str, contexts_path: Path | str, output_path: Path | str) -> None:
    cards_path = Path(cards_path)
    contexts_path = Path(contexts_path)
    output_path = Path(output_path)
    context_by_trace = {row["trace_id"]: row for row in _load_jsonl(contexts_path)}
    card_rows = _load_jsonl(cards_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in card_rows:
            if row.get("status") != "ok" or row.get("decision") == "reject":
                continue
            trace_id = row["trace_id"]
            context = context_by_trace.get(trace_id, {})
            for idx, card in enumerate(row.get("question_cards", [])):
                card_id = _stable_id(trace_id, str(idx), card.get("query", ""), prefix="hhcard__")
                review_id = f"review__{card_id}"
                review_result = {
                    "decision": "",
                    "notes": "",
                }
                for field in REVIEW_BOOL_FIELDS:
                    review_result[field] = None
                task = HardHallucinationReviewTask(
                    review_id=review_id,
                    card_id=card_id,
                    trace_id=trace_id,
                    knowledge_base_category=row.get("knowledge_base_category", ""),
                    query=card.get("query", ""),
                    context=context.get("context_text", ""),
                    hard_hallucination_family=card.get("hard_hallucination_family", ""),
                    target_failure_mode=card.get("target_failure_mode", ""),
                    target_failure_subtype=card.get("target_failure_subtype", ""),
                    trigger_mechanism_label=card.get("trigger_mechanism_label", ""),
                    expected_safe_behavior=card.get("expected_safe_behavior", ""),
                    why_this_is_hallucination=card.get("why_this_is_hallucination", ""),
                    evidence_source_hint=card.get("evidence_source_hint", []),
                    judge_anchor=card.get("judge_anchor", ""),
                    review_result=review_result,
                )
                handle.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build review tasks for hard hallucination question cards")
    parser.add_argument("--cards", default=DEFAULT_CARDS)
    parser.add_argument("--contexts", default=DEFAULT_CONTEXTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_review_tasks(args.cards, args.contexts, args.output)
    print(f"Wrote review tasks to: {args.output}")


if __name__ == "__main__":
    main()
