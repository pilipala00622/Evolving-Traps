#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from llm import LLM


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def build_prompt(
    seed: Dict[str, Any],
    gene: Dict[str, Any],
    context_record: Dict[str, Any],
    variants: int,
    max_context_chars: int,
    profile: str = "general",
) -> str:
    context_text = context_record["context_text"]
    if len(context_text) > max_context_chars:
        context_text = context_text[:max_context_chars]
    payload = {
        "seed_id": seed["seed_id"],
        "pattern_name": seed.get("pattern_name"),
        "knowledge_base_category": seed.get("knowledge_base_category"),
        "source_query": seed.get("source_query"),
        "mechanism": seed.get("mechanism"),
        "target_error_type": gene.get("target_error_type"),
        "task_frame": gene.get("task_frame"),
        "failure_mechanism": gene.get("failure_mechanism"),
        "trigger_form": gene.get("trigger_form"),
        "support_gap_type": gene.get("support_gap_type"),
        "answer_carrier": gene.get("answer_carrier"),
        "abstention_expected": gene.get("abstention_expected"),
        "difficulty_knobs": gene.get("difficulty_knobs", []),
        "verifier_shape": gene.get("verifier_shape"),
        "mutation_axes": gene.get("mutation_axes", []),
        "correct_behavior": seed.get("correct_behavior"),
    }
    profile_extra = ""
    if profile == "numeric_fabrication":
        profile_extra = """
8. 本轮是 numeric_fabrication 专项：
   - 每个候选题都必须是精确数值问题，且 `answer_carrier` 固定为 `numeric`
   - 每个候选题都必须维持 `target_error_type = 无中生有`
   - 必须显式要求一个精确数值输出（如保留两位小数、给出精确金额/利润/比例）
   - 但文档中必须故意缺失一个关键计算口径、隐藏系数、专属调整项或不可替代的规则参数
   - 可以加入相关行业基准、可比数值、相似统计口径作为强干扰，但这些不能足以支撑精确计算
   - `expected_good_behavior` 必须明确写成：文档不足以支持精确数值，不得估算或拍脑袋补全
9. 不允许把问题退化成判断题，也不允许让正确行为变成普通抽取。
"""

    return f"""你是一个 hard hallucination benchmark writer。请基于给定 gene 和文档上下文，生成 {variants} 个不同方向但同一机制的候选题。

硬约束：
1. 只生成 B组主池风格的问题：专业边界、证据不足、背景当直接证据。
2. 不允许开放创作，不允许写续写、感悟、风格模仿。
3. 必须让“文档不足时应保留/拒答/限定回答”成为正确行为。
4. 每个候选题都必须是单目标、可判定、可做后续小规模人工判别。
5. 候选题不能直接复述原始 source_query，要做受控变体。
6. 输出为 JSON array，每个元素必须包含：
   - variant_id
   - query
   - intended_failure_mechanism
   - target_error_type
   - answer_carrier
   - expected_good_behavior
   - verifier_hint
   - difficulty_knob_used
7. `intended_failure_mechanism` 必须等于 gene 中的 `failure_mechanism`。
8. `target_error_type` 必须使用这三个标签之一：`无中生有`、`越权推理`、`生成错误`。
{profile_extra}

gene 与 seed：
{json.dumps(payload, ensure_ascii=False, indent=2)}

文档上下文（可能截断）：
{context_text}
"""


def normalize_candidate(seed: Dict[str, Any], gene: Dict[str, Any], context_record: Dict[str, Any], card: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    variant_id = card.get("variant_id", "v")
    gene_suffix = (gene.get("gene_id") or gene.get("seed_id") or "gene").split("_")[-1]
    return {
        "candidate_id": f"{seed['seed_id']}__{gene_suffix}__{variant_id}",
        "variant_id": variant_id,
        "gene_id": gene.get("gene_id"),
        "seed_id": seed["seed_id"],
        "trace_id": seed.get("trace_id"),
        "knowledge_base_category": seed.get("knowledge_base_category"),
        "source_query": seed.get("source_query"),
        "source_question_text": context_record.get("source_question_text"),
        "context_text": context_record.get("context_text"),
        "context_length": context_record.get("context_length"),
        "query": card.get("query"),
        "intended_failure_mechanism": card.get("intended_failure_mechanism"),
        "target_error_type": card.get("target_error_type"),
        "answer_carrier": card.get("answer_carrier"),
        "expected_good_behavior": card.get("expected_good_behavior"),
        "verifier_hint": card.get("verifier_hint"),
        "difficulty_knob_used": card.get("difficulty_knob_used"),
        "task_frame": gene.get("task_frame"),
        "support_gap_type": gene.get("support_gap_type"),
        "verifier_shape": gene.get("verifier_shape"),
        "model_name": model_name,
    }


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_gene(
    seed: Dict[str, Any],
    gene: Dict[str, Any],
    context_record: Dict[str, Any],
    model_name: str,
    variants: int,
    max_context_chars: int,
    profile: str = "general",
) -> List[Dict[str, Any]]:
    llm = LLM(model_name)
    response = llm.get_model_answer(
        build_prompt(
            seed,
            gene,
            context_record,
            variants=variants,
            max_context_chars=max_context_chars,
            profile=profile,
        )
    )
    payload = extract_json_payload(response)
    if isinstance(payload, dict):
        payload = [payload]
    return [normalize_candidate(seed, gene, context_record, card, model_name) for card in payload]


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand normalized genes into candidate hard hallucination questions.")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--genes", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--variants-per-gene", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-context-chars", type=int, default=14000)
    parser.add_argument("--profile", default="general")
    args = parser.parse_args()

    seeds = {rec["seed_id"]: rec for rec in read_jsonl(Path(args.seeds))}
    genes = [rec for rec in read_jsonl(Path(args.genes)) if "error" not in rec]
    contexts = {rec["trace_id"]: rec for rec in read_jsonl(Path(args.contexts))}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    jobs = []
    for gene in genes:
        seed = seeds[gene["seed_id"]]
        context_record = contexts[seed["trace_id"]]
        jobs.append((seed, gene, context_record))

    all_records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_gene,
                seed,
                gene,
                context_record,
                args.model,
                args.variants_per_gene,
                args.max_context_chars,
                args.profile,
            ): (seed, gene)
            for seed, gene, context_record in jobs
        }
        for future in as_completed(futures):
            seed, gene = futures[future]
            try:
                records = future.result()
                for record in records:
                    append_jsonl(output_path, record)
                    all_records.append(record)
                print(f"[ok] {seed['seed_id']} -> {len(records)} variants", flush=True)
            except Exception as exc:
                error_record = {
                    "seed_id": seed["seed_id"],
                    "pattern_name": seed.get("pattern_name"),
                    "trace_id": seed.get("trace_id"),
                    "model_name": args.model,
                    "error": str(exc),
                }
                append_jsonl(output_path, error_record)
                all_records.append(error_record)
                print(f"[error] {seed['seed_id']}: {exc}", flush=True)


if __name__ == "__main__":
    main()
