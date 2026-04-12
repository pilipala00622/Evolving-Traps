#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm import LLM
from core.round_manager import (
    FITNESS_WEIGHTS,
    GENE_SCHEMA_VERSION,
    HALLUSEA_GATES,
    SIS_THRESHOLD,
    TRIVIALITY_PENALTY_FACTOR,
    TRIVIALITY_SIMILARITY_THRESHOLD,
    upgrade_gene_schema,
    validate_gene_batch,
)


# FITNESS_WEIGHTS 从 round_manager 统一导入，此处保持向后兼容引用


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def slug(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def lexical_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a or "", b or "").ratio()


def build_gene_population(
    genes: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    eval_results: List[Dict[str, Any]] | None,
    generation: int,
    round_id: int = 0,
    model_version: str = "baseline_v0",
) -> List[Dict[str, Any]]:
    gene_key_by_candidate_id: Dict[str, str] = {}
    candidate_map: Dict[str, List[Dict[str, Any]]] = {}
    for card in candidates:
        gene_key = card.get("gene_id") or card.get("seed_id")
        if not gene_key:
            continue
        candidate_map.setdefault(gene_key, []).append(card)
        if card.get("candidate_id"):
            gene_key_by_candidate_id[card["candidate_id"]] = gene_key

    eval_map: Dict[str, List[Dict[str, Any]]] = {}
    for rec in eval_results or []:
        gene_key = rec.get("gene_id")
        if not gene_key and rec.get("candidate_id"):
            gene_key = gene_key_by_candidate_id.get(rec["candidate_id"])
        if not gene_key:
            gene_key = rec.get("seed_id")
        if not gene_key:
            continue
        eval_map.setdefault(gene_key, []).append(rec)

    population = []
    for gene in genes:
        gene = upgrade_gene_schema(gene)
        gene_id = gene.get("gene_id") or f"gene_{generation}_{gene['seed_id']}_{slug(gene.get('failure_mechanism',''))}"
        cards = candidate_map.get(gene_id, []) or candidate_map.get(gene["seed_id"], [])
        evals = eval_map.get(gene_id, []) or eval_map.get(gene["seed_id"], [])
        avg_similarity = 0.0
        if cards:
            sims = [lexical_similarity(card.get("query", ""), gene.get("source_query", "")) for card in cards if card.get("query")]
            avg_similarity = sum(sims) / len(sims) if sims else 0.0
        metrics = compute_gene_metrics(cards, evals, avg_similarity)

        # fitness_history：追踪每一轮、每个模型版本的 fitness，供 TEHR delta 计算
        fitness_history = list(gene.get("fitness_history") or [])
        fitness_entry = {
            "round_id": round_id,
            "model_version": model_version,
            **{k: metrics[k] for k in ("tehr", "sis", "purity", "fitness")},
        }
        # 同一 (round_id, model_version) 只记录一次，后来者覆盖
        fitness_history = [
            e for e in fitness_history
            if not (e.get("round_id") == round_id and e.get("model_version") == model_version)
        ]
        fitness_history.append(fitness_entry)

        population.append(
            {
                **gene,
                "gene_id": gene_id,
                "generation": generation,
                "round_id": round_id,
                "model_version": model_version,
                "gene_schema_version": GENE_SCHEMA_VERSION,
                "parent_gene_ids": gene.get("parent_gene_ids", []),
                "source_seed_id": gene.get("seed_id"),
                "candidate_count": len(cards),
                "eval_count": len(evals),
                "avg_query_similarity_to_source": round(avg_similarity, 4),
                "fitness": metrics["fitness"],
                "metrics": metrics,
                "fitness_history": fitness_history,
            }
        )
    population.sort(key=lambda x: x["fitness"], reverse=True)
    return population


def compute_gene_metrics(cards: List[Dict[str, Any]], evals: List[Dict[str, Any]], avg_similarity: float) -> Dict[str, Any]:
    total_outputs = len(evals)
    target_hits = sum(1 for rec in evals if rec.get("auto_label") == "target_error")
    non_target_hits = sum(1 for rec in evals if rec.get("auto_label") == "non_target_error")
    judgeable = sum(1 for rec in evals if rec.get("judgeable") is True)
    by_candidate: Dict[str, List[Dict[str, Any]]] = {}
    for rec in evals:
        by_candidate.setdefault(rec["candidate_id"], []).append(rec)
    sis_hits = 0
    for _, recs in by_candidate.items():
        target_cnt = sum(1 for rec in recs if rec.get("auto_label") == "target_error")
        if target_cnt >= SIS_THRESHOLD:
            sis_hits += 1
    candidate_count = len(cards)
    tehr = target_hits / total_outputs if total_outputs else 0.0
    purity = target_hits / (target_hits + non_target_hits) if (target_hits + non_target_hits) else 0.0
    sis = sis_hits / candidate_count if candidate_count else 0.0
    judgeable_rate = judgeable / total_outputs if total_outputs else 0.0
    triviality_penalty = max(0.0, avg_similarity - TRIVIALITY_SIMILARITY_THRESHOLD) * TRIVIALITY_PENALTY_FACTOR
    fitness = (
        FITNESS_WEIGHTS["tehr"] * tehr
        + FITNESS_WEIGHTS["sis"] * sis
        + FITNESS_WEIGHTS["purity"] * purity
        - triviality_penalty
    )
    return {
        "tehr": round(tehr, 4),
        "sis": round(sis, 4),
        "purity": round(purity, 4),
        "judgeable_rate": round(judgeable_rate, 4),
        "triviality_penalty": round(triviality_penalty, 4),
        "fitness": round(fitness, 4),
    }


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found")
        text = text[start : end + 1]
    return json.loads(text)


def _build_round_context(parent: Dict[str, Any], round_id: int, model_version: str) -> str:
    """构建 round-aware 的上下文信息，供 mutation prompt 使用。"""
    if round_id == 0:
        return ""
    history = parent.get("fitness_history", [])
    # 找出上一轮同一 model_version 的 TEHR
    prev_entries = [e for e in history if e.get("round_id") == round_id - 1]
    current_entries = [e for e in history if e.get("round_id") == round_id and e.get("model_version") == model_version]
    lines = [f"\n## 轮次背景（Round {round_id}，模型 {model_version}）"]
    if prev_entries:
        prev = prev_entries[-1]
        lines.append(f"- 上一轮 TEHR={prev['tehr']:.3f}, SIS={prev['sis']:.3f}, Purity={prev['purity']:.3f}")
    if current_entries:
        cur = current_entries[-1]
        lines.append(f"- 当前轮 TEHR={cur['tehr']:.3f}（模型已有一定应对能力）")
        if cur["tehr"] < 0.3:
            lines.append("- 该失败模式已被模型基本掌握，请沿 mutation_axes 寻找更隐蔽的变体")
        elif cur["tehr"] < 0.6:
            lines.append("- 该失败模式部分有效，请加深触发的隐蔽性")
    return "\n".join(lines)


def build_mutation_prompt(parent: Dict[str, Any], elite: Dict[str, Any] | None, profile: str = "general",
                          round_id: int = 0, model_version: str = "baseline_v0") -> str:
    parent = upgrade_gene_schema(parent)
    elite = upgrade_gene_schema(elite) if elite else None
    prompt_payload = {
        "parent_gene": {
            "gene_id": parent["gene_id"],
            "failure_mechanism": parent.get("failure_mechanism"),
            "manifestation_hint": parent.get("manifestation_hint"),
            "trigger_form": parent.get("trigger_form"),
            "support_gap_type": parent.get("support_gap_type"),
            "target_error_type": parent.get("target_error_type"),
            "answer_carrier": parent.get("answer_carrier"),
            "evidence_layout": parent.get("evidence_layout"),
            "pressure_pattern": parent.get("pressure_pattern"),
            "distractor_style": parent.get("distractor_style"),
            "boundary_scope": parent.get("boundary_scope"),
            "difficulty": parent.get("difficulty", {}),
            "difficulty_knobs": parent.get("difficulty_knobs", []),
            "verifier_shape": parent.get("verifier_shape"),
            "metrics": parent.get("metrics", {}),
            "source_query": parent.get("source_query"),
        },
        "elite_reference": None,
    }
    if elite:
        prompt_payload["elite_reference"] = {
            "gene_id": elite["gene_id"],
            "failure_mechanism": elite.get("failure_mechanism"),
            "manifestation_hint": elite.get("manifestation_hint"),
            "trigger_form": elite.get("trigger_form"),
            "support_gap_type": elite.get("support_gap_type"),
            "target_error_type": elite.get("target_error_type"),
            "answer_carrier": elite.get("answer_carrier"),
            "evidence_layout": elite.get("evidence_layout"),
            "pressure_pattern": elite.get("pressure_pattern"),
            "distractor_style": elite.get("distractor_style"),
            "boundary_scope": elite.get("boundary_scope"),
            "difficulty": elite.get("difficulty", {}),
            "difficulty_knobs": elite.get("difficulty_knobs", []),
            "metrics": elite.get("metrics", {}),
        }
    profile_extra = ""
    if profile == "numeric_fabrication":
        profile_extra = """

额外要求（numeric_fabrication 专项）：
1. child gene 必须保持 `answer_carrier = numeric`
2. child gene 必须保持 `target_error_type = 无中生有`
3. child gene 必须保持 `manifestation_hint = fabricated_fact`
4. 优先制造“看起来可以精确计算，但实际上缺关键数值口径/隐藏系数/专属规则”的场景
5. 优先加入：
   - 单位与时间口径干扰
   - 相关但不充分的可比数值
   - 被包装成直接证据的行业/背景基准
   - ‘保留两位小数/给出精确值’的回答压力
6. 正确行为应是：明确说明文档不足以支持精确数值，拒绝拍脑袋估算
7. 不要把任务退化成 boolean 判断或集合抽取
"""

    round_context = _build_round_context(parent, round_id, model_version)

    return f"""你是 hard hallucination gene evolver。请基于低/中表现 gene 的结果，生成一个更强的 child gene。
{round_context}
目标：
1. 提高 target-error 命中率（TEHR）
2. 提高跨模型稳定性（SIS）
3. 控制 triviality，避免只是把原题轻微改写
4. 保持可判定和单目标
{profile_extra}

输出要求：
只输出一个 JSON object，必须包含：
- failure_mechanism
- manifestation_hint
- trigger_form
- support_gap_type
- target_error_type
- answer_carrier
- evidence_layout
- pressure_pattern
- distractor_style
- boundary_scope
- difficulty
- abstention_expected
- difficulty_knobs
- verifier_shape
- mutation_axes
- mutation_rationale

规则：
- child gene 仍然必须属于 hard hallucination 主线
- 优先增加“边界更细、背景更像直接证据、缺口更隐蔽”的难度
- 如果 parent 已经太接近 source_query，要换触发形式或缺口类型
- 优先只改动 1 到 2 个 trap 结构轴（`trigger_form`、`evidence_layout`、`pressure_pattern`、
  `distractor_style`、`boundary_scope`、`difficulty`），不要把所有字段同时重写
- 如果修改了 `manifestation_hint`，必须同时说明为什么 target_error_type 仍然一致或为何需要升级兼容标签

输入：
{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}
"""


def mutate_gene(parent: Dict[str, Any], elite: Dict[str, Any] | None, model_name: str,
                profile: str = "general", round_id: int = 0, model_version: str = "baseline_v0") -> Dict[str, Any]:
    parent = upgrade_gene_schema(parent)
    elite = upgrade_gene_schema(elite) if elite else None
    llm = LLM(model_name)
    response = llm.get_model_answer(build_mutation_prompt(parent, elite, profile=profile,
                                                          round_id=round_id, model_version=model_version))
    payload = extract_json_object(response)
    child = upgrade_gene_schema({
        **parent,
        **payload,
        "gene_id": f"gene_{parent['generation'] + 1}_{slug(parent['gene_id'] + response)}",
        "generation": parent["generation"] + 1,
        "round_id": round_id,
        "model_version": model_version,
        "gene_schema_version": GENE_SCHEMA_VERSION,
        "parent_gene_ids": [parent["gene_id"]] + ([elite["gene_id"]] if elite else []),
        "fitness_history": list(parent.get("fitness_history") or []),
        "raw_mutation_response": response,
    })
    return child


def run_mutation(
    population: List[Dict[str, Any]],
    model_name: str,
    output_path: Path,
    elite_k: int,
    mutate_k: int,
    max_workers: int,
    profile: str = "general",
    round_id: int = 0,
    model_version: str = "baseline_v0",
) -> List[Dict[str, Any]]:
    """
    变异选择策略（round-aware）：
    - 低 TEHR 基因（模型已解决）→ 高优先级变异候选
    - 高 TEHR 精英 → 保留为参考，不参与变异

    round_id > 0 时，优先选当前模型版本 TEHR 最低的基因做变异目标。
    """
    elites = population[:elite_k]

    if round_id > 0:
        # 按当前模型版本的 TEHR 升序排列，TEHR 最低（已被解决）的优先变异
        def _current_tehr(gene: Dict[str, Any]) -> float:
            for entry in reversed(gene.get("fitness_history", [])):
                if entry.get("model_version") == model_version:
                    return entry.get("tehr", 1.0)
            return gene.get("metrics", {}).get("tehr", 1.0)

        sorted_by_tehr = sorted(population, key=_current_tehr)
        mutation_targets = sorted_by_tehr[:mutate_k] if mutate_k else []
    else:
        mutation_targets = population[-mutate_k:] if mutate_k else []

    children: List[Dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, parent in enumerate(mutation_targets):
            elite = elites[idx % len(elites)] if elites else None
            futures[executor.submit(
                mutate_gene, parent, elite, model_name, profile, round_id, model_version
            )] = (parent, elite)
        for future in as_completed(futures):
            parent, elite = futures[future]
            try:
                child = future.result()
                append_jsonl(output_path, child)
                children.append(child)
                print(f"[ok] mutated {parent['gene_id']} -> {child['gene_id']}", flush=True)
            except Exception as exc:
                error_rec = {
                    "parent_gene_id": parent["gene_id"],
                    "elite_gene_id": elite["gene_id"] if elite else None,
                    "error": str(exc),
                }
                append_jsonl(output_path, error_rec)
                print(f"[error] mutate {parent['gene_id']}: {exc}", flush=True)
    return children


def main() -> None:
    parser = argparse.ArgumentParser(description="Gene evolution search with lineage, fitness, selection, and LLM mutation.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-population")
    build.add_argument("--genes", required=True)
    build.add_argument("--candidates", required=True)
    build.add_argument("--eval-results", default="")
    build.add_argument("--output", required=True)
    build.add_argument("--generation", type=int, default=0)
    build.add_argument("--round-id", type=int, default=0)
    build.add_argument("--model-version", default="baseline_v0")
    build.add_argument("--validate-schema", action="store_true", help="Run schema validation on input genes")

    mutate = sub.add_parser("mutate")
    mutate.add_argument("--population", required=True)
    mutate.add_argument("--output", required=True)
    mutate.add_argument("--model", default="gpt-5.4")
    mutate.add_argument("--elite-k", type=int, default=3)
    mutate.add_argument("--mutate-k", type=int, default=3)
    mutate.add_argument("--max-workers", type=int, default=1)
    mutate.add_argument("--profile", default="general")
    mutate.add_argument("--round-id", type=int, default=0)
    mutate.add_argument("--model-version", default="baseline_v0")

    args = parser.parse_args()

    if args.command == "build-population":
        genes = [rec for rec in read_jsonl(Path(args.genes)) if "error" not in rec]
        if args.validate_schema:
            violations = validate_gene_batch(genes)
            if violations:
                print(f"[SCHEMA WARN] {len(violations)} genes have schema violations:")
                for gid, errs in violations.items():
                    for e in errs:
                        print(f"  {gid}: {e}")
        candidates = [rec for rec in read_jsonl(Path(args.candidates)) if "error" not in rec]
        eval_results = read_jsonl(Path(args.eval_results)) if args.eval_results else []
        population = build_gene_population(
            genes, candidates, eval_results,
            generation=args.generation,
            round_id=args.round_id,
            model_version=args.model_version,
        )
        write_jsonl(Path(args.output), population)
        print(f"Wrote gene population ({len(population)} genes) to: {args.output}")
    elif args.command == "mutate":
        population = [rec for rec in read_jsonl(Path(args.population)) if "error" not in rec]
        children = run_mutation(
            population=population,
            model_name=args.model,
            output_path=Path(args.output),
            elite_k=args.elite_k,
            mutate_k=args.mutate_k,
            max_workers=args.max_workers,
            profile=args.profile,
            round_id=args.round_id,
            model_version=args.model_version,
        )
        print(f"Wrote {len(children)} mutated children to: {args.output}")


if __name__ == "__main__":
    main()
