#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

from llm import LLM


VISIBLE_COT_PATTERNS = [
    re.compile(r"<think>(.*?)</think>\s*(.*)", re.S),
    re.compile(r"<thinking>(.*?)</thinking>\s*(.*)", re.S),
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_questions(record: Dict[str, Any], question_field: str) -> Iterable[tuple[int, str]]:
    raw = record.get(question_field)
    if isinstance(raw, list):
        for idx, question in enumerate(raw):
            if isinstance(question, str) and question.strip():
                yield idx, question.strip()
        return
    if isinstance(raw, str) and raw.strip():
        yield 0, raw.strip()


def extract_visible_cot(answer: str) -> Dict[str, str]:
    text = (answer or "").strip()
    for pattern in VISIBLE_COT_PATTERNS:
        match = pattern.fullmatch(text)
        if match:
            return {
                "visible_reasoning": match.group(1).strip(),
                "final_answer": match.group(2).strip(),
                "capture_mode": "visible_reasoning_split",
            }
    return {
        "visible_reasoning": "",
        "final_answer": text,
        "capture_mode": "final_only",
    }


def build_jobs(records: List[Dict[str, Any]], question_field: str, limit: int) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for record in records:
        for question_index, question in normalize_questions(record, question_field):
            item_id = f"{record.get('prompt_sn', 'item')}__q{question_index}"
            jobs.append(
                {
                    "item_id": item_id,
                    "prompt_sn": record.get("prompt_sn", ""),
                    "question_index": question_index,
                    "question": question,
                    "meta": {k: v for k, v in record.items() if k != question_field},
                }
            )
            if limit and len(jobs) >= limit:
                return jobs
    return jobs


def collect_one(job: Dict[str, Any], model_name: str, preserve_visible_cot: bool) -> Dict[str, Any]:
    llm = LLM(model_name)
    response_text = llm.get_model_answer(job["question"])
    parsed = extract_visible_cot(response_text) if preserve_visible_cot else {
        "visible_reasoning": "",
        "final_answer": (response_text or "").strip(),
        "capture_mode": "final_only",
    }
    return {
        "item_id": job["item_id"],
        "prompt_sn": job["prompt_sn"],
        "question_index": job["question_index"],
        "question": job["question"],
        "model_name": model_name,
        "response_text": response_text,
        "final_answer": parsed["final_answer"],
        "visible_reasoning": parsed["visible_reasoning"],
        "reasoning_capture_mode": parsed["capture_mode"],
        "hidden_cot_preserved": False,
        "trace_id": getattr(llm, "trace_id", ""),
        "usage_info": getattr(llm, "usage_info", {}) or {},
        "timing_info": getattr(llm, "timing_info", {}) or {},
        "source_meta": job["meta"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect multi-model answers for a benchmark JSONL and preserve only visible reasoning, not hidden CoT."
    )
    parser.add_argument("--input", required=True, help="Path to benchmark JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--models", nargs="+", required=True, help="Model list")
    parser.add_argument("--question-field", default="questions", help="Field containing question text or question list")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on flattened question count")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--preserve-visible-cot",
        action="store_true",
        help="Split visible reasoning markers like <think>...</think>; hidden/internal CoT is never captured.",
    )
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    jobs = build_jobs(records, args.question_field, args.limit)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    futures = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for job in jobs:
            for model_name in args.models:
                future = executor.submit(collect_one, job, model_name, args.preserve_visible_cot)
                futures[future] = (job["item_id"], model_name)

        for future in as_completed(futures):
            item_id, model_name = futures[future]
            try:
                append_jsonl(output_path, future.result())
                print(f"[ok] {item_id} @ {model_name}", flush=True)
            except Exception as exc:
                append_jsonl(
                    output_path,
                    {
                        "item_id": item_id,
                        "model_name": model_name,
                        "error": str(exc),
                    },
                )
                print(f"[error] {item_id} @ {model_name}: {exc}", flush=True)


if __name__ == "__main__":
    main()
