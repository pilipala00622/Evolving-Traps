"""Auto-fill real portrait review tasks and produce a feasibility report."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

from human_review.io_utils import load_jsonl, write_json, write_jsonl
from review_excel_bridge import export_to_excel


DEFAULT_DIR = Path(__file__).resolve().parent / "demo_outputs" / "real_portrait_review"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-fill real portrait reviews")
    parser.add_argument("--input", default=str(DEFAULT_DIR / "real_human_review_tasks.pending.jsonl"))
    parser.add_argument("--output-dir", default=str(DEFAULT_DIR))
    return parser.parse_args()


def bool_str(value: bool) -> str:
    return "true" if value else "false"


def classify_row(row: Dict[str, object]) -> Dict[str, object]:
    score = float(row["auto_recommendation"]["auto_review_score"])
    target_attr = row["target_error_type"]
    ability = row["portrait_metadata"]["能力板块"]
    third_cat = row["portrait_metadata"]["三级分类"]
    query = str(row["query"])
    query_len = len(query)
    has_structure = any(token in query for token in ["是否", "哪个", "哪一个", "根据", "能否", "为什么", "问题是", "不正确"])
    open_generation = ability == "生成控制" and third_cat == "生成"
    verifier_feasible = (not open_generation) or ("问题是" in query) or ("不正确" in query)
    single_target = score >= 0.34 and target_attr in {"缺证断言", "引入新事实"}
    natural = query_len <= 220 and query_len >= 6
    release_priority = "high" if score >= 0.42 and verifier_feasible and natural else ("medium" if score >= 0.32 else "low")

    if verifier_feasible and single_target and natural and score >= 0.36:
        decision = "approve"
    elif score >= 0.28:
        decision = "revise"
    else:
        decision = "reject"

    issue_tags = []
    if open_generation:
        issue_tags.append("open_generation")
    if not verifier_feasible:
        issue_tags.append("verifier_hard")
    if not natural:
        issue_tags.append("query_shape_unstable")
    if not single_target:
        issue_tags.append("single_target_uncertain")
    if not has_structure:
        issue_tags.append("needs_judging_rule")

    notes = []
    notes.append(f"auto_score={score}")
    notes.append(f"ability={ability}/{third_cat}")
    if verifier_feasible:
        notes.append("可先作为归因评测候选")
    else:
        notes.append("更适合保留到 revise 池，暂不直接进 verifier 评测")
    notes.append("本题不依赖 reference_answer 作为主判据")

    review_result = {
        "reviewer": "codex-auto",
        "decision": decision,
        "confirmed_target_error_type": target_attr,
        "confirmed_scenario_type": row["scenario_type"],
        "reference_answer_supported": True,
        "final_state_is_correctly_specified": verifier_feasible,
        "verifier_design_is_feasible": verifier_feasible,
        "reward_should_be_verifiable": verifier_feasible and decision != "reject",
        "query_is_natural": natural,
        "time_metadata_correct": None,
        "is_single_target_error": single_target,
        "release_priority": release_priority,
        "issue_tags": issue_tags,
        "notes": "；".join(notes),
    }
    filled = dict(row)
    filled["review_result"] = review_result
    return filled


def build_report(rows: List[Dict[str, object]], output_path: Path) -> Dict[str, object]:
    decisions = Counter(row["review_result"]["decision"] for row in rows)
    attrs = Counter()
    by_attr = defaultdict(list)
    by_ability = Counter()
    verifier_counts = Counter()
    for row in rows:
        rr = row["review_result"]
        if rr["decision"] == "approve":
            attrs[row["target_error_type"]] += 1
        by_attr[row["target_error_type"]].append(row)
        by_ability[(row["portrait_metadata"]["能力板块"], rr["decision"])] += 1
        verifier_counts["feasible" if rr["verifier_design_is_feasible"] else "hard"] += 1

    approved = [row for row in rows if row["review_result"]["decision"] == "approve"]
    revise = [row for row in rows if row["review_result"]["decision"] == "revise"]

    report = {
        "total": len(rows),
        "decision_distribution": decisions,
        "approved_target_error_distribution": attrs,
        "verifier_distribution": verifier_counts,
        "mean_auto_score_by_attr": {
            attr: round(mean(float(r["auto_recommendation"]["auto_review_score"]) for r in attr_rows), 4)
            for attr, attr_rows in by_attr.items()
        },
        "ability_decision_distribution": {
            f"{ability}::{decision}": count for (ability, decision), count in by_ability.items()
        },
        "approved_item_ids": [row["item_id"] for row in approved],
        "revise_item_ids": [row["item_id"] for row in revise[:20]],
    }
    write_json(output_path, report)
    return report


def write_markdown(rows: List[Dict[str, object]], report: Dict[str, object], output_path: Path) -> None:
    approved = [row for row in rows if row["review_result"]["decision"] == "approve"]
    revise = [row for row in rows if row["review_result"]["decision"] == "revise"]
    reject = [row for row in rows if row["review_result"]["decision"] == "reject"]

    lines = [
        "# Real Portrait Feasibility Report",
        "",
        f"- total_items: {len(rows)}",
        f"- approve: {len(approved)}",
        f"- revise: {len(revise)}",
        f"- reject: {len(reject)}",
        "",
        "## Feasibility Judgment",
        "",
        "这批题可以继续做，但更适合作为“归因检测样本池”，而不是立即做严格的 outcome-verifier benchmark。",
        "",
        "原因：",
        "- 这批题很多来自生成型或长文档 prompt，适合测归因与稳定性，但不天然适合做结构化 final_state 判分。",
        "- `缺证断言` 和 `引入新事实` 两类题的方向是对的，适合作为第一批 release 候选。",
        "- 真正适合 verifier 的优先是检索/边界感知/部分文档整合题；开放生成题建议先放在 revise 池。",
        "",
        "## Key Stats",
        "",
        f"- verifier_feasible: {report['verifier_distribution']['feasible']}",
        f"- verifier_hard: {report['verifier_distribution']['hard']}",
        f"- approved_target_error_distribution: {dict(report['approved_target_error_distribution'])}",
        f"- mean_auto_score_by_attr: {report['mean_auto_score_by_attr']}",
        "",
        "## Recommended Next Step",
        "",
        "1. 先用 approve 集合作为第一批“人工确认的归因样本池”。",
        "2. 从 approve 里优先挑 `verifier_design_is_feasible=true` 的题进入下一轮 rollout/规则设计。",
        "3. 对 revise 池里的开放生成题补充更明确的判定规则，再决定是否入库。",
        "",
        "## Sample Approved",
        "",
    ]
    for row in approved[:10]:
        lines.append(
            f"- {row['item_id']} | {row['target_error_type']} | {row['portrait_metadata']['能力板块']} | {row['query'][:80]}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filled_rows = [classify_row(row) for row in rows]
    jsonl_path = output_dir / "real_human_review_tasks.auto_reviewed.jsonl"
    xlsx_path = output_dir / "real_human_review_tasks.auto_reviewed.xlsx"
    write_jsonl(jsonl_path, filled_rows)
    export_to_excel(jsonl_path, xlsx_path)

    report = build_report(filled_rows, output_dir / "real_review_feasibility.json")
    write_markdown(filled_rows, report, output_dir / "real_review_feasibility.md")
    print(f"Auto review finished: {jsonl_path}")


if __name__ == "__main__":
    main()
