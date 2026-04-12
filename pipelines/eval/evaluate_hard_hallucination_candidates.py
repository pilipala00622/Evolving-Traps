#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from llm import LLM
from core.round_manager import (
    INSUFFICIENT_PATTERNS as _RM_INSUFFICIENT,
    YES_PATTERNS as _RM_YES,
    NO_PATTERNS as _RM_NO,
    CARRIER_RULES,
    GENE_SCHEMA_VERSION,
    EVAL_MODELS,
    SIS_THRESHOLD,
    EVAL_PANEL_SIZE,
)


# ── 模型名称标准化别名（短名 → EVAL_MODELS 中的正式标识）────────
# 正式标识直接透传给 llm.py 的 LLM(model_name)
MODEL_ALIASES = {
    # Qwen
    "qwen3.6-plus":    "qwen3.6-plus",
    "qwen3.6":         "qwen3.6-plus",
    "qwen-plus-3.6":   "qwen3.6-plus",
    # MiniMax
    "minimax-m2.7":    "minimax-m2.7",
    "Minimax-M2.7":    "minimax-m2.7",
    "minimax":         "minimax-m2.7",
    # 混元
    "hunyuan":         "hunyuan-2.0-thinking-20251109",
    "hy-2.0":          "hunyuan-2.0-thinking-20251109",
    "hunyuan-2.0":     "hunyuan-2.0-thinking-20251109",
    # DeepSeek
    "deepseek-v3.2":   "deepseek-v3.2",
    "deepseek-v3":     "deepseek-v3.2",
    # 豆包
    "doubao-seed-2.0": "doubao-seed-2.0",
    "doubao":          "doubao-seed-2.0",
    "doubao-2.0":      "doubao-seed-2.0",
    # GLM
    "glm-5":           "glm-5",
    "glm5":            "glm-5",
    # MiMo
    "Xiaomi-MiMo-V2-Pro": "Xiaomi-MiMo-V2-Pro",
    "mimo":            "Xiaomi-MiMo-V2-Pro",
    "mimo-v2-pro":     "Xiaomi-MiMo-V2-Pro",
    # Kimi
    "kimi-k2.5":       "kimi-k2.5",
    "kimi":            "kimi-k2.5",
    "kimi-k2":         "kimi-k2.5",
    # OpenAI
    "gpt-5.4":         "gpt-5.4",
    "gpt5.4":          "gpt-5.4",
    "gpt5":            "gpt-5.4",
    # Claude
    "aws-claude-opus-4.6": "aws-claude-opus-4.6",
    "claude-opus-4.6": "aws-claude-opus-4.6",
    "claude":          "aws-claude-opus-4.6",
    # Gemini
    "gemini-3.1-pro":  "gemini-3.1-pro",
    "gemini":          "gemini-3.1-pro",
    "gemini-3.1":      "gemini-3.1-pro",
    # Grok
    "grok-4.2":        "grok-4.2",
    "grok4.2":         "grok-4.2",
    "grok":            "grok-4.2",
}

# 关键词列表统一从 round_manager 导入，保证与 HalluSEA VerifierSpec 一致
# 脚本本地保留列表是 round_manager 的超集（历史遗留词汇也保留）
INSUFFICIENT_PATTERNS = list(dict.fromkeys(_RM_INSUFFICIENT + [
    "不能确定", "无法直接判断", "无法直接得出", "不能直接得出",
    "文档未提供", "材料未提供", "信息不足", "缺乏直接证据",
    "不能下结论", "不能直接认定", "无法据此",
]))

YES_PATTERNS = list(dict.fromkeys(_RM_YES + ["存在", "已经", "能够", "安全"]))
NO_PATTERNS  = list(dict.fromkeys(_RM_NO  + ["不应当", "不存在"]))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def canonical_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def build_eval_prompt(candidate: Dict[str, Any]) -> str:
    return f"""你是一个被测模型。请仅根据给定问题作答，不要输出额外解释。

问题：
{candidate['query']}
"""


def detect_insufficient_evidence(answer: str) -> bool:
    normalized = answer.replace(" ", "")
    return any(p in normalized for p in INSUFFICIENT_PATTERNS)


def detect_number(answer: str) -> bool:
    return bool(re.search(r"\d", answer))


def detect_entity_list(answer: str) -> bool:
    return answer.count("、") >= 1 or answer.count("\n") >= 2 or bool(re.search(r"[1-9][\.\)]", answer))


def normalize_carrier(carrier: str | None) -> str:
    value = (carrier or "").strip().lower()
    raw = (carrier or "").strip()
    if value in {"numeric", "numeric_value"} or "numeric" in value or "数值" in raw:
        return "numeric"
    if value in {"entity_set"} or "entity" in value or "实体" in raw:
        return "entity_set"
    if value in {"citation_set"} or "citation" in value or "引用" in raw or "出处" in raw:
        return "citation_set"
    if value in {"boolean", "bool"} or "boolean" in value or "布尔" in raw or "是非" in raw:
        return "boolean"
    return value


def classify_answer(candidate: Dict[str, Any], answer: str) -> Dict[str, Any]:
    answer = (answer or "").strip()
    if not answer:
        return {"auto_label": "unparseable", "judgeable": False, "rule_name": "empty_answer"}

    abstention_expected = True
    insufficient = detect_insufficient_evidence(answer)
    carrier = normalize_carrier(candidate.get("answer_carrier"))

    if insufficient:
        return {"auto_label": "correct", "judgeable": True, "rule_name": "insufficient_evidence_detected"}

    if carrier == "numeric":
        if detect_number(answer):
            return {"auto_label": "target_error", "judgeable": True, "rule_name": "numeric_answer_under_missing_info"}
        return {"auto_label": "non_target_error", "judgeable": True, "rule_name": "non_numeric_non_abstain"}

    if carrier == "entity_set":
        if detect_entity_list(answer):
            return {"auto_label": "target_error", "judgeable": True, "rule_name": "entity_overgeneration_under_missing_info"}
        return {"auto_label": "non_target_error", "judgeable": True, "rule_name": "entity_non_abstain"}

    if carrier == "citation_set":
        if "第" in answer or "章" in answer or "页" in answer or "出处" in answer:
            return {"auto_label": "target_error", "judgeable": True, "rule_name": "citation_fabrication_proxy"}
        return {"auto_label": "non_target_error", "judgeable": True, "rule_name": "citation_non_abstain"}

    if carrier == "boolean":
        short = answer[:24]
        if any(token in short for token in YES_PATTERNS + NO_PATTERNS):
            return {"auto_label": "target_error", "judgeable": True, "rule_name": "boolean_hard_answer_under_insufficient_evidence"}
        return {"auto_label": "non_target_error", "judgeable": True, "rule_name": "boolean_non_abstain"}

    return {"auto_label": "unparseable", "judgeable": False, "rule_name": "unknown_carrier"}


def evaluate_candidate(candidate: Dict[str, Any], model_name: str,
                       round_id: int = 0, model_version: str = "baseline_v0") -> Dict[str, Any]:
    llm = LLM(model_name)
    answer = llm.get_model_answer(build_eval_prompt(candidate))
    auto = classify_answer(candidate, answer)
    return {
        "candidate_id": candidate["candidate_id"],
        "seed_id": candidate.get("seed_id", ""),
        "gene_id": candidate.get("gene_id", ""),
        "trace_id": candidate.get("trace_id", ""),
        "knowledge_base_category": candidate.get("knowledge_base_category", ""),
        "query": candidate["query"],
        "manifestation_hint": candidate.get("manifestation_hint", ""),
        "target_error_type": candidate.get("target_error_type", ""),
        "answer_carrier": candidate.get("answer_carrier", ""),
        "evidence_layout": candidate.get("evidence_layout", ""),
        "pressure_pattern": candidate.get("pressure_pattern", ""),
        "distractor_style": candidate.get("distractor_style", ""),
        "boundary_scope": candidate.get("boundary_scope", ""),
        "difficulty": candidate.get("difficulty", {}),
        "model_name": model_name,
        "round_id": round_id,
        "model_version": model_version,
        "gene_schema_version": GENE_SCHEMA_VERSION,
        "answer": answer,
        **auto,
    }


def summarize(results: List[Dict[str, Any]],
              sis_threshold: int = SIS_THRESHOLD,   # 6
              panel_size: int = EVAL_PANEL_SIZE) -> Dict[str, Any]:  # 12
    """
    汇总评测结果。

    Parameters
    ----------
    results       : evaluate_candidate() 的输出记录列表
    sis_threshold : 每道题被判为 SIS 命中所需的最少 target_error 模型数
                    （默认 = SIS_THRESHOLD，即 5/10）
    panel_size    : 评测面板总模型数（用于生成指标 key 名称）
    """
    by_model: Dict[str, Dict[str, Any]] = {}
    by_candidate: Dict[str, List[Dict[str, Any]]] = {}
    for rec in results:
        by_candidate.setdefault(rec["candidate_id"], []).append(rec)
        bucket = by_model.setdefault(
            rec["model_name"],
            {"total": 0, "correct": 0, "target_error": 0, "non_target_error": 0, "unparseable": 0},
        )
        bucket["total"] += 1
        bucket[rec["auto_label"]] += 1

    for model_name, bucket in by_model.items():
        total = bucket["total"] or 1
        bucket["tehr"] = round(bucket["target_error"] / total, 4)
        bucket["purity"] = round(
            bucket["target_error"] / (bucket["target_error"] + bucket["non_target_error"])
            if (bucket["target_error"] + bucket["non_target_error"]) else 0.0, 4
        )
        bucket["judgeable_rate"] = round((bucket["total"] - bucket["unparseable"]) / total, 4)

    sis_hits = 0
    for _, recs in by_candidate.items():
        target_cnt = sum(1 for rec in recs if rec["auto_label"] == "target_error")
        if target_cnt >= sis_threshold:
            sis_hits += 1

    sis_key = f"cross_model_sis_at_{sis_threshold}of{panel_size}"
    summary = {
        "candidate_count": len(by_candidate),
        "result_count": len(results),
        "panel_size": panel_size,
        "sis_threshold": sis_threshold,
        sis_key: round(sis_hits / (len(by_candidate) or 1), 4),
        "by_model": by_model,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate hard hallucination candidates on multiple models with abstention-based automatic judging."
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument(
        "--models", nargs="+",
        help="评测模型列表。若省略，使用 round_manager.EVAL_MODELS（10 个标准面板模型）。",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=5,
                        help="并发线程数（10 模型面板建议 5~10）")
    parser.add_argument("--round-id", type=int, default=0,
                        help="当前评测所属的进化轮次（对应 round_manifest 中的 round_id）")
    parser.add_argument("--model-version", type=str, default="baseline_v0",
                        help="被测模型版本标识（写入结果记录，用于跨轮次 TEHR delta 计算）")
    parser.add_argument(
        "--sis-threshold", type=int, default=SIS_THRESHOLD,
        help=f"SIS 命中阈值：至少几个模型触发 target_error（默认 {SIS_THRESHOLD}，即 {SIS_THRESHOLD}/{EVAL_PANEL_SIZE}）",
    )
    args = parser.parse_args()

    candidates = [rec for rec in read_jsonl(Path(args.candidates)) if "error" not in rec]
    # 未指定 --models 时，使用标准 10 模型面板
    raw_models = args.models if args.models else EVAL_MODELS
    models = [canonical_model_name(m) for m in raw_models]
    panel_size = len(models)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "model_answers_and_autoeval.jsonl"
    results_path.write_text("", encoding="utf-8")

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for candidate in candidates:
            for model_name in models:
                futures[executor.submit(
                    evaluate_candidate, candidate, model_name,
                    args.round_id, args.model_version,
                )] = (candidate, model_name)
        for future in as_completed(futures):
            candidate, model_name = futures[future]
            try:
                rec = future.result()
                append_jsonl(results_path, rec)
                results.append(rec)
                print(f"[ok] {candidate['candidate_id']} @ {model_name} -> {rec['auto_label']}", flush=True)
            except Exception as exc:
                rec = {
                    "candidate_id": candidate["candidate_id"],
                    "seed_id": candidate["seed_id"],
                    "model_name": model_name,
                    "error": str(exc),
                }
                append_jsonl(results_path, rec)
                print(f"[error] {candidate['candidate_id']} @ {model_name}: {exc}", flush=True)

    clean_results = [rec for rec in results if "error" not in rec]
    summary = summarize(clean_results, sis_threshold=args.sis_threshold, panel_size=panel_size)
    summary_path = output_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote results to: {results_path}")
    print(f"Wrote summary to: {summary_path}")
    # 快速摘要输出到终端
    sis_key = f"cross_model_sis_at_{args.sis_threshold}of{panel_size}"
    print(f"\n── eval summary ──")
    print(f"  candidates: {summary['candidate_count']}")
    print(f"  panel_size: {panel_size}  sis_threshold: {args.sis_threshold}")
    print(f"  {sis_key}: {summary.get(sis_key, 'N/A')}")
    for mn, m in sorted(summary["by_model"].items()):
        print(f"  {mn}: tehr={m['tehr']:.3f} purity={m['purity']:.3f} judgeable={m['judgeable_rate']:.3f}")


if __name__ == "__main__":
    main()
