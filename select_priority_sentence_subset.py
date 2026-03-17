"""Select a small high-priority subset from sentence annotation tasks."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from human_review.io_utils import load_jsonl, write_json, write_jsonl
from response_sentence_excel_bridge import export_to_excel


DEFAULT_INPUT = Path(__file__).resolve().parent / "demo_outputs" / "response_sentence_annotations" / "sentence_annotation_tasks.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "demo_outputs" / "response_sentence_annotations" / "priority_subset"


ASSERTIVE_HINTS = [
    "是", "属于", "位于", "发生于", "导致", "说明", "表明", "可以", "必须", "已经", "就是", "应当",
]
RISK_HINTS = [
    "根据文档", "可以看出", "因此", "所以", "由此", "说明", "表明", "建议", "应", "可", "能够",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a high-priority subset for sentence annotation")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--subset-size", type=int, default=60)
    parser.add_argument("--per-response-cap", type=int, default=8)
    return parser.parse_args()


def compute_priority(row: Dict[str, object]) -> Tuple[float, Dict[str, float]]:
    sentence = str(row.get("sentence_text", "") or "")
    intended = str(row.get("intended_failure_mode", "") or "")
    prefill_h = row.get("prefill_is_hallucinated", None)
    prefill_attr = str(row.get("prefill_attribution_type", "") or "")

    length_score = min(len(sentence) / 80.0, 1.0)
    assertive_score = 1.0 if any(token in sentence for token in ASSERTIVE_HINTS) else 0.0
    reasoning_score = 1.0 if any(token in sentence for token in RISK_HINTS) else 0.0
    digit_score = 1.0 if re.search(r"\d", sentence) else 0.0
    intended_bonus = 1.0 if intended in {"缺证断言", "引入新事实"} else 0.4
    prefill_bonus = 1.0 if (prefill_h is True or prefill_attr) else 0.0

    score = (
        0.24 * length_score
        + 0.22 * assertive_score
        + 0.18 * reasoning_score
        + 0.12 * digit_score
        + 0.12 * intended_bonus
        + 0.12 * prefill_bonus
    )
    return round(score, 4), {
        "length_score": round(length_score, 4),
        "assertive_score": round(assertive_score, 4),
        "reasoning_score": round(reasoning_score, 4),
        "digit_score": round(digit_score, 4),
        "intended_bonus": round(intended_bonus, 4),
        "prefill_bonus": round(prefill_bonus, 4),
    }


def select_subset(rows: List[Dict[str, object]], subset_size: int, per_response_cap: int) -> List[Dict[str, object]]:
    scored = []
    for row in rows:
        score, breakdown = compute_priority(row)
        enriched = dict(row)
        enriched["priority_score"] = score
        enriched["priority_breakdown"] = breakdown
        scored.append(enriched)

    scored.sort(key=lambda row: row["priority_score"], reverse=True)

    picked: List[Dict[str, object]] = []
    per_response_counts = defaultdict(int)
    covered_queries = set()

    # First pass: ensure breadth across queries.
    for row in scored:
        if len(picked) >= subset_size:
            break
        query_id = row.get("query_id", "")
        response_id = row.get("response_id", "")
        if query_id in covered_queries:
            continue
        if per_response_counts[response_id] >= per_response_cap:
            continue
        picked.append(row)
        covered_queries.add(query_id)
        per_response_counts[response_id] += 1

    # Second pass: fill remaining slots by pure priority with per-response cap.
    for row in scored:
        if len(picked) >= subset_size:
            break
        response_id = row.get("response_id", "")
        annotation_id = row.get("annotation_id", "")
        if per_response_counts[response_id] >= per_response_cap:
            continue
        if any(existing.get("annotation_id") == annotation_id for existing in picked):
            continue
        picked.append(row)
        per_response_counts[response_id] += 1

    return picked


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset = select_subset(rows, args.subset_size, args.per_response_cap)
    jsonl_path = output_dir / "sentence_annotation_tasks.priority.jsonl"
    xlsx_path = output_dir / "sentence_annotation_tasks.priority.xlsx"
    write_jsonl(jsonl_path, subset)
    export_to_excel(jsonl_path, xlsx_path)

    summary = {
        "total_source_sentences": len(rows),
        "selected_sentences": len(subset),
        "query_count": len({row.get("query_id", "") for row in subset}),
        "response_count": len({row.get("response_id", "") for row in subset}),
        "mean_priority_score": round(sum(row["priority_score"] for row in subset) / max(len(subset), 1), 4),
        "top_examples": [
            {
                "annotation_id": row.get("annotation_id", ""),
                "model_name": row.get("model_name", ""),
                "query_id": row.get("query_id", ""),
                "priority_score": row.get("priority_score", 0.0),
                "sentence_text": str(row.get("sentence_text", ""))[:120],
            }
            for row in subset[:10]
        ],
    }
    write_json(output_dir / "priority_subset_summary.json", summary)


if __name__ == "__main__":
    main()
