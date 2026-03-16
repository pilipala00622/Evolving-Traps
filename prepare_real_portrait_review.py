"""Prepare a human-review workbook from real phase-1 question portraits."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from human_review.io_utils import write_json, write_jsonl
from review_excel_bridge import export_to_excel


DEFAULT_SOURCE = Path(__file__).resolve().parent / "第一阶段-14个模型-gpt51评测" / "question_portraits_detailed.jsonl"
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "demo_outputs" / "real_portrait_review"
TARGET_ATTRS = {"缺证断言", "引入新事实"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare real portrait review files")
    parser.add_argument("--input", default=str(DEFAULT_SOURCE), help="question_portraits_detailed.jsonl path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTDIR), help="output directory")
    parser.add_argument("--max-items", type=int, default=60, help="max rows to export")
    parser.add_argument(
        "--target-attributions",
        nargs="+",
        default=sorted(TARGET_ATTRS),
        help="attribution labels to keep",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def compute_priority(row: Dict[str, object]) -> float:
    avg_rate = float(row.get("平均幻觉率", 0.0))
    discr = float(row.get("区分度得分", 0.0))
    attr_dist = row.get("归因分布", {}) or {}
    total = sum(attr_dist.values()) or 1
    top = max(attr_dist.values()) if attr_dist else 0
    clarity = top / total if total else 0.0
    return round(0.4 * avg_rate + 0.35 * discr + 0.25 * clarity, 4)


def extract_short_question(raw_question: str) -> str:
    if not raw_question:
        return ""
    match = re.search(r"\[问题\]\s*[:：]\s*(.+)$", raw_question, flags=re.S)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in raw_question.splitlines() if line.strip()]
    return lines[-1] if lines else raw_question[:200]


def build_review_row(row: Dict[str, object]) -> Dict[str, object]:
    raw_question = row.get("问题", "")
    short_question = extract_short_question(raw_question)
    reference_answer = row.get("参考答案") or "【源数据缺失：第一阶段画像文件未提供参考答案】"
    return {
        "item_id": row.get("prompt_id", ""),
        "target_error_type": row.get("主导归因类型", ""),
        "scenario_type": "static",
        "query": short_question,
        "context_preview": str(raw_question)[:3000],
        "reference_answer": reference_answer,
        "plan_id": f"{row.get('建议_domain') or row.get('二级分类') or 'unknown'}__{row.get('主导归因类型','unknown')}__portrait",
        "auto_signals": {
            "target_error_trigger_rate": row.get("平均幻觉率", 0.0),
            "non_target_error_leakage": 1.0 - min(1.0, max((max((row.get('归因分布', {}) or {}).values(), default=0) / max(sum((row.get('归因分布', {}) or {}).values()), 1)), 0.0)),
            "dominant_error_match_rate": 1.0,
            "answerability_rate": 1.0,
            "naturalness_mean": 4.0,
            "anchor_score_mean": round((1.0 - float(row.get("平均幻觉率", 0.0))) * 100, 2),
            "anchor_difficulty_bucket": "medium",
        },
        "required_checks": [
            "确认该题是否真的主要打在目标归因类型上",
            "如果后续补充了参考答案，再确认答案是否与题目/文档支撑一致",
            "确认是否适合作为后续可验证 benchmark 候选",
            "确认 query 是否自然、是否值得进入 release 池",
        ],
        "role_responsibilities": {},
        "review_result": {
            "reviewer": "",
            "decision": "",
            "confirmed_target_error_type": row.get("主导归因类型", ""),
            "confirmed_scenario_type": "static",
            "reference_answer_supported": None,
            "final_state_is_correctly_specified": None,
            "verifier_design_is_feasible": None,
            "reward_should_be_verifiable": None,
            "query_is_natural": None,
            "time_metadata_correct": None,
            "is_single_target_error": None,
            "release_priority": "medium",
            "issue_tags": [],
            "notes": "",
        },
        "auto_recommendation": {
            "auto_review_score": compute_priority(row),
            "suggested_decision": "approve" if compute_priority(row) >= 0.3 else "revise",
            "reasons": [
                f"avg_hallucination={row.get('平均幻觉率', 0.0):.3f}",
                f"discrimination={row.get('区分度得分', 0.0):.3f}",
                f"dominant_attr={row.get('主导归因类型', '')}",
            ],
        },
        "portrait_metadata": {
            "一级分类": row.get("一级分类"),
            "二级分类": row.get("二级分类"),
            "三级分类": row.get("三级分类"),
            "能力板块": row.get("能力板块"),
            "主导原始错误类型": row.get("主导原始错误类型"),
            "归因分布": row.get("归因分布"),
            "模型数": row.get("模型数"),
            "模型列表": row.get("模型列表"),
            "高幻觉模型": row.get("高幻觉模型"),
            "低幻觉模型": row.get("低幻觉模型"),
            "区分度得分": row.get("区分度得分"),
            "平均幻觉率": row.get("平均幻觉率"),
        },
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_attrs = set(args.target_attributions)
    rows = [row for row in read_jsonl(input_path) if row.get("主导归因类型") in target_attrs]
    rows.sort(key=compute_priority, reverse=True)
    selected = rows[: args.max_items]

    review_rows = [build_review_row(row) for row in selected]
    jsonl_path = output_dir / "real_human_review_tasks.pending.jsonl"
    xlsx_path = output_dir / "real_human_review_tasks.pending.xlsx"
    write_jsonl(jsonl_path, review_rows)
    export_to_excel(jsonl_path, xlsx_path)

    summary = {
        "input_path": str(input_path),
        "export_count": len(review_rows),
        "target_error_distribution": Counter(row["target_error_type"] for row in review_rows),
        "top_prompt_ids": [row["item_id"] for row in review_rows[:10]],
    }
    write_json(output_dir / "real_review_summary.json", summary)
    print(f"Prepared real portrait review xlsx: {xlsx_path}")


if __name__ == "__main__":
    main()
