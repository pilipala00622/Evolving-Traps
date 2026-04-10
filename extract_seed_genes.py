#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from llm import LLM


GENE_FIELDS = [
    "seed_id",
    "task_frame",
    "failure_mechanism",
    "trigger_form",
    "support_gap_type",
    "target_error_type",
    "answer_carrier",
    "abstention_expected",
    "difficulty_knobs",
    "verifier_shape",
    "mutation_axes",
    "non_triviality_reason",
    "confidence_notes",
]


def load_seed_cards(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") not in {"locked", "keep", "approved"}:
                continue
            records.append(rec)
    return records


def build_prompt(seed: Dict[str, Any]) -> str:
    seed_payload = {
        "seed_id": seed["seed_id"],
        "pattern_name": seed.get("pattern_name"),
        "knowledge_base_category": seed.get("knowledge_base_category"),
        "source_query": seed.get("source_query"),
        "mechanism": seed.get("mechanism"),
        "user_intent": seed.get("user_intent"),
        "conflict_point": seed.get("conflict_point"),
        "correct_behavior": seed.get("correct_behavior"),
        "target_error_type": seed.get("target_error_type"),
        "answer_carrier": seed.get("answer_carrier"),
        "abstention_expected": seed.get("abstention_expected"),
        "difficulty_knobs": seed.get("difficulty_knobs", []),
        "why_likely_to_fail": seed.get("why_likely_to_fail"),
    }
    return f"""你是一个严格的 benchmark gene extractor。请从下面的人类 seed 卡片中抽取“可复用的诱导基因（gene）”，用于后续自动扩写 hard hallucination 题目。

要求：
1. 只输出一个 JSON object，不要输出解释。
2. 字段必须完整，且必须包含这些键：
{json.dumps(GENE_FIELDS, ensure_ascii=False)}
3. 所有字段都要尽量简洁、可复用，不要复述整条原题。
4. `difficulty_knobs` 和 `mutation_axes` 必须是字符串数组。
5. `task_frame` 只从这些值中选择一个：
   - boundary_judgment
   - constrained_extraction
   - constrained_reasoning
   - citation_localization
6. `support_gap_type` 只从这些值中选择一个或两个最主要的，若两个就用 `+` 连接：
   - missing_direct_evidence
   - missing_key_variable
   - background_not_decision
   - rule_to_case_gap
   - special_population_gap
   - incomplete_case_facts
7. `verifier_shape` 只从这些值中选择一个：
   - boolean_boundary_check
   - numeric_sufficiency_check
   - entity_overgeneration_check
   - citation_fabrication_check
8. `trigger_form` 用短语描述问题触发形式，例如：
   - yes_no_boundary_question
   - forced_numeric_estimation
   - case_rule_application
   - constrained_summary_request

人类 seed：
{json.dumps(seed_payload, ensure_ascii=False, indent=2)}
"""


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model output")
        text = text[start : end + 1]
    obj = json.loads(text)
    return obj


def normalize_gene(seed: Dict[str, Any], gene: Dict[str, Any], model_name: str, raw_response: str) -> Dict[str, Any]:
    rec = {k: gene.get(k) for k in GENE_FIELDS}
    rec["seed_id"] = seed["seed_id"]
    rec["trace_id"] = seed.get("trace_id")
    rec["pattern_name"] = seed.get("pattern_name")
    rec["knowledge_base_category"] = seed.get("knowledge_base_category")
    rec["source_query"] = seed.get("source_query")
    rec["mechanism"] = seed.get("mechanism")
    rec["source_answer_carrier"] = seed.get("answer_carrier")
    rec["source_target_error_type"] = seed.get("target_error_type")
    rec["model_name"] = model_name
    rec["raw_response"] = raw_response
    if not isinstance(rec.get("difficulty_knobs"), list):
        rec["difficulty_knobs"] = []
    if not isinstance(rec.get("mutation_axes"), list):
        rec["mutation_axes"] = []
    return rec


def process_seed(seed: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    llm = LLM(model_name)
    prompt = build_prompt(seed)
    response = llm.get_model_answer(prompt)
    gene = extract_json_object(response)
    return normalize_gene(seed, gene, model_name=model_name, raw_response=response)


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract reusable genes from human-written seed cards.")
    parser.add_argument("--input", required=True, help="Path to seed cards JSONL")
    parser.add_argument("--output", required=True, help="Path to output gene bank JSONL")
    parser.add_argument("--model", default="gpt-5.4", help="LLM model name")
    parser.add_argument("--max-workers", type=int, default=3, help="Concurrent workers")
    args = parser.parse_args()

    seed_cards = load_seed_cards(Path(args.input))
    if not seed_cards:
        raise SystemExit("No locked/approved seed cards found.")

    results: List[Dict[str, Any]] = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_seed, seed, args.model): seed for seed in seed_cards}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                record = future.result()
                results.append(record)
                append_jsonl(output_path, record)
                print(f"[ok] {seed['seed_id']}", flush=True)
            except Exception as exc:
                record = {
                    "seed_id": seed["seed_id"],
                    "pattern_name": seed.get("pattern_name"),
                    "knowledge_base_category": seed.get("knowledge_base_category"),
                    "source_query": seed.get("source_query"),
                    "model_name": args.model,
                    "error": str(exc),
                }
                results.append(record)
                append_jsonl(output_path, record)
                print(f"[error] {seed['seed_id']}: {exc}", flush=True)

    results.sort(key=lambda x: x["seed_id"])
    write_jsonl(output_path, results)


if __name__ == "__main__":
    main()
