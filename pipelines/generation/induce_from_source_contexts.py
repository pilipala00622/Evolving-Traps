#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from llm import LLM
from core.round_manager import upgrade_gene_schema


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    buffer = ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() and not buffer:
                continue
            buffer += line
            try:
                rows.append(json.loads(buffer))
                buffer = ""
            except json.JSONDecodeError:
                continue
    if buffer.strip():
        raise ValueError(f"Incomplete JSON object while reading {path}")
    return rows


def extract_json_payload(text: str) -> Any:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start_arr = text.find("[")
        end_arr = text.rfind("]")
        start_obj = text.find("{")
        end_obj = text.rfind("}")
        if start_arr != -1 and end_arr > start_arr:
            text = text[start_arr : end_arr + 1]
        elif start_obj != -1 and end_obj > start_obj:
            text = text[start_obj : end_obj + 1]
        else:
            raise ValueError("No JSON payload found in model output")
    return json.loads(text)


def build_prompt(manifest: dict[str, Any], context_record: dict[str, Any], variants: int, max_context_chars: int) -> str:
    manifest = upgrade_gene_schema(manifest)
    context_text = context_record["context_text"]
    if len(context_text) > max_context_chars:
        context_text = context_text[:max_context_chars]

    gene_payload = {
        "gene_id": manifest["gene_id"],
        "seed_id": manifest["seed_id"],
        "knowledge_base_category": manifest["knowledge_base_category"],
        "target_error_type": manifest["target_error_type"],
        "manifestation_hint": manifest.get("manifestation_hint"),
        "answer_carrier": manifest["answer_carrier"],
        "task_frame": manifest["task_frame"],
        "failure_mechanism": manifest["failure_mechanism"],
        "support_gap_type": manifest["support_gap_type"],
        "evidence_layout": manifest.get("evidence_layout"),
        "pressure_pattern": manifest.get("pressure_pattern"),
        "distractor_style": manifest.get("distractor_style"),
        "boundary_scope": manifest.get("boundary_scope"),
        "difficulty": manifest.get("difficulty", {}),
        "verifier_shape": manifest["verifier_shape"],
        "seed_pattern_name": manifest["seed_pattern_name"],
        "seed_source_query": manifest["seed_source_query"],
        "seed_correct_behavior": manifest["seed_correct_behavior"],
    }

    return f"""你是一个 hard hallucination benchmark writer。请把给定 gene 迁移到一个新的 source context 上，生成 {variants} 个可判定、单目标的诱导题。

要求：
1. 只能生成 B 组主池风格的问题：边界判断、证据不足、背景当直接证据、缺失关键信息下硬答。
2. 如果当前 context 与 gene 机制不兼容，直接输出空数组 `[]`。
3. 每个候选题必须保留 gene 的：
   - `target_error_type`
   - `manifestation_hint`
   - `answer_carrier`
   - `failure_mechanism`
4. 候选题必须基于当前 context 的事实边界重写，不能复述旧 seed query。
5. 正确行为必须仍然是保留、拒答、限定回答，或明确说明证据不足。
6. 输出 JSON array；每个元素必须包含：
   - variant_id
   - query
   - intended_failure_mechanism
   - manifestation_hint
   - target_error_type
   - answer_carrier
   - expected_good_behavior
   - verifier_hint
   - evidence_layout
   - pressure_pattern
   - distractor_style
   - boundary_scope
   - design_difficulty
   - difficulty_knob_used

gene：
{json.dumps(gene_payload, ensure_ascii=False, indent=2)}

新 context 的原始 query 信息：
{json.dumps({
    'source_trace_id': manifest['source_trace_id'],
    'source_query_category': manifest.get('source_query_category'),
    'source_query': manifest.get('source_query'),
    'source_question_text': manifest.get('source_question_text'),
}, ensure_ascii=False, indent=2)}

文档上下文（可能截断）：
{context_text}
"""


def normalize_candidate(manifest: dict[str, Any], card: dict[str, Any], model_name: str) -> dict[str, Any]:
    manifest = upgrade_gene_schema(manifest)
    variant_id = card.get("variant_id", "var_001")
    return {
        "candidate_id": f"{manifest['seed_id']}__{manifest['gene_id'].split('_')[-1]}__{manifest['source_trace_id']}__{variant_id}",
        "variant_id": variant_id,
        "gene_id": manifest["gene_id"],
        "seed_id": manifest["seed_id"],
        "trace_id": manifest["source_trace_id"],
        "knowledge_base_category": manifest["knowledge_base_category"],
        "source_query": manifest.get("source_query"),
        "source_question_text": manifest.get("source_question_text"),
        "query": card.get("query"),
        "intended_failure_mechanism": card.get("intended_failure_mechanism"),
        "manifestation_hint": card.get("manifestation_hint", manifest.get("manifestation_hint")),
        "target_error_type": card.get("target_error_type", manifest.get("target_error_type")),
        "answer_carrier": card.get("answer_carrier", manifest.get("answer_carrier")),
        "expected_good_behavior": card.get("expected_good_behavior"),
        "verifier_hint": card.get("verifier_hint"),
        "difficulty_knob_used": card.get("difficulty_knob_used"),
        "evidence_layout": card.get("evidence_layout", manifest.get("evidence_layout")),
        "pressure_pattern": card.get("pressure_pattern", manifest.get("pressure_pattern")),
        "distractor_style": card.get("distractor_style", manifest.get("distractor_style")),
        "boundary_scope": card.get("boundary_scope", manifest.get("boundary_scope")),
        "difficulty": card.get("design_difficulty", manifest.get("difficulty")),
        "task_frame": manifest.get("task_frame"),
        "support_gap_type": manifest.get("support_gap_type"),
        "verifier_shape": manifest.get("verifier_shape"),
        "model_name": model_name,
        "induction_mode": "source_context_transfer",
        "source_gene_id": manifest["gene_id"],
        "source_seed_trace_id": manifest.get("seed_trace_id"),
        "source_manifest_id": manifest["manifest_id"],
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_manifest(manifest: dict[str, Any], context_record: dict[str, Any], model_name: str, variants: int, max_context_chars: int) -> list[dict[str, Any]]:
    llm = LLM(model_name)
    response = llm.get_model_answer(build_prompt(manifest, context_record, variants=variants, max_context_chars=max_context_chars))
    payload = extract_json_payload(response)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Expected JSON array payload")
    return [normalize_candidate(manifest, card, model_name) for card in payload]


def main() -> None:
    parser = argparse.ArgumentParser(description="Induce new candidates from source contexts using archived genes.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--variants-per-pair", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=14000)
    args = parser.parse_args()

    manifest_rows = read_jsonl(Path(args.manifest))
    contexts = {row["trace_id"]: row for row in read_jsonl(Path(args.contexts))}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for row in manifest_rows:
            context_record = contexts[row["source_trace_id"]]
            futures[executor.submit(process_manifest, row, context_record, args.model, args.variants_per_pair, args.max_context_chars)] = row

        for future in as_completed(futures):
            manifest = futures[future]
            try:
                records = future.result()
                for record in records:
                    append_jsonl(output_path, record)
                print(f"[ok] {manifest['manifest_id']} -> {len(records)} variants", flush=True)
            except Exception as exc:
                append_jsonl(
                    output_path,
                    {
                        "manifest_id": manifest["manifest_id"],
                        "gene_id": manifest["gene_id"],
                        "seed_id": manifest["seed_id"],
                        "trace_id": manifest["source_trace_id"],
                        "model_name": args.model,
                        "error": str(exc),
                    },
                )
                print(f"[error] {manifest['manifest_id']}: {exc}", flush=True)


if __name__ == "__main__":
    main()
