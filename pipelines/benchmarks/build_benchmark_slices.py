#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from core.round_manager import SIS_THRESHOLD, EVAL_PANEL_SIZE


DEFAULT_ROUND_SOURCES = [
    (
        "round0",
        Path("data/hard_hallucination/approved_round0_candidates.jsonl"),
        Path("data/hard_hallucination/eval_round0_doubao_fast9_qwen_hy_gemini/model_answers_and_autoeval.jsonl"),
    ),
    (
        "round1",
        Path("data/hard_hallucination/candidate_cards_round1_doubao_children.jsonl"),
        Path("data/hard_hallucination/eval_round1_children_qwen_hy_gemini/model_answers_and_autoeval.jsonl"),
    ),
    (
        "round2",
        Path("data/hard_hallucination/candidate_cards_round2_archive_children.jsonl"),
        Path("data/hard_hallucination/eval_round2_archive_children_qwen_hy_gemini_v2/model_answers_and_autoeval.jsonl"),
    ),
    (
        "round3",
        Path("data/hard_hallucination/candidate_cards_round3_archive_children_v2.jsonl"),
        Path("data/hard_hallucination/eval_round3_archive_children_qwen_hy_gemini_v2/model_answers_and_autoeval.jsonl"),
    ),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def normalize_carrier(carrier: str | None) -> str:
    value = (carrier or "").strip().lower()
    raw = (carrier or "").strip()
    if value in {"numeric", "numeric_value"} or "numeric" in value or "数值" in raw:
        return "numeric"
    if value in {"entity_set"} or "entity" in value or "实体" in raw:
        return "entity_set"
    if value in {"citation_set"} or "citation" in value or "引用" in raw:
        return "citation_set"
    if value in {"boolean", "bool"} or "boolean" in value or "布尔" in raw or "是非" in raw:
        return "boolean"
    return value or "unknown"


def candidate_metrics(
    candidate: dict[str, Any],
    eval_rows: list[dict[str, Any]],
    round_name: str,
    sis_threshold: int = SIS_THRESHOLD,
) -> dict[str, Any]:
    target = sum(1 for r in eval_rows if r.get("auto_label") == "target_error")
    non_target = sum(1 for r in eval_rows if r.get("auto_label") == "non_target_error")
    correct = sum(1 for r in eval_rows if r.get("auto_label") == "correct")
    unparseable = sum(1 for r in eval_rows if r.get("auto_label") == "unparseable")
    total = len(eval_rows)
    purity = target / (target + non_target) if (target + non_target) else 0.0
    tehr = target / total if total else 0.0
    # SIS@6/12：至少 sis_threshold（默认 6）个模型触发 target_error
    sis_6of12 = 1 if target >= sis_threshold else 0
    discriminative = 1 if (target >= 1 and (non_target >= 1 or correct >= 1)) else 0
    score = round(0.5 * sis_6of12 + 0.3 * purity + 0.2 * tehr, 4)
    return {
        **candidate,
        "source_round": round_name,
        "normalized_answer_carrier": normalize_carrier(candidate.get("answer_carrier")),
        "eval_total": total,
        "target_error_count": target,
        "non_target_error_count": non_target,
        "correct_count": correct,
        "unparseable_count": unparseable,
        "tehr": round(tehr, 4),
        "purity": round(purity, 4),
        "sis_6of12": sis_6of12,                   # 主指标（6/12 面板）
        "sis_threshold_used": sis_threshold,
        "discriminative_flag": discriminative,
        "model_labels": {r["model_name"]: r["auto_label"] for r in eval_rows},
        "selection_score": score,
    }


def build_candidate_table(round_sources: list[tuple[str, Path, Path]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for round_name, candidate_path, eval_path in round_sources:
        candidates = {r["candidate_id"]: r for r in read_jsonl(candidate_path)}
        eval_rows = [r for r in read_jsonl(eval_path) if "error" not in r]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in eval_rows:
            grouped[row["candidate_id"]].append(row)
        for candidate_id, rows in grouped.items():
            if candidate_id not in candidates:
                continue
            table.append(candidate_metrics(candidates[candidate_id], rows, round_name))
    table.sort(key=lambda r: (r["selection_score"], r["target_error_count"], r["purity"]), reverse=True)
    return table


def choose_main_slice(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 12 模型面板配额（boolean 最多 5，其余各 3/3/2）
    quotas = {"boolean": 5, "numeric": 3, "entity_set": 3, "citation_set": 2}
    selected: list[dict[str, Any]] = []
    seen = set()
    for carrier, quota in quotas.items():
        eligible = [
            r
            for r in rows
            if r["normalized_answer_carrier"] == carrier
            and r["sis_6of12"] == 1        # 主指标：6/12 面板命中
            and r["purity"] >= 0.66
            and r["candidate_id"] not in seen
        ]
        for rec in eligible[:quota]:
            selected.append(rec)
            seen.add(rec["candidate_id"])
    selected.sort(key=lambda r: (r["normalized_answer_carrier"], -r["selection_score"]))
    return selected


def choose_diagnostic_slice(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        r
        for r in rows
        if r["discriminative_flag"] == 1 and r["purity"] >= 0.66
    ]
    selected.sort(key=lambda r: (r["normalized_answer_carrier"], -r["selection_score"]))
    return selected[:8]


def build_report(main_rows: list[dict[str, Any]], diagnostic_rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    sis_thresh = summary.get("sis_threshold", SIS_THRESHOLD)
    panel = summary.get("eval_panel_size", EVAL_PANEL_SIZE)
    lines = [
        "# Benchmark Slice Report v1",
        "",
        "## Summary",
        f"- candidate_table_count: {summary['candidate_table_count']}",
        f"- main_slice_count: {summary['main_slice_count']}",
        f"- diagnostic_slice_count: {summary['diagnostic_slice_count']}",
        f"- eval_panel_size: {panel}",
        f"- sis_threshold: {sis_thresh}/{panel}  (SIS@{sis_thresh}of{panel})",
        "",
        "## Main Slice",
    ]
    for rec in main_rows:
        lines.append(
            f"- `{rec['candidate_id']}` | round={rec['source_round']} | carrier={rec['normalized_answer_carrier']}"
            f" | type={rec['target_error_type']} | TEHR={rec['tehr']} | Purity={rec['purity']}"
            f" | SIS_6of12={rec['sis_6of12']}"
        )
    lines += ["", "## Diagnostic Slice"]
    for rec in diagnostic_rows:
        lines.append(
            f"- `{rec['candidate_id']}` | round={rec['source_round']} | carrier={rec['normalized_answer_carrier']}"
            f" | type={rec['target_error_type']} | target_hits={rec['target_error_count']}/{rec['eval_total']}"
            f" | labels={rec['model_labels']}"
        )
    return "\n".join(lines) + "\n"


def parse_round_sources(raw_values: list[str] | None) -> list[tuple[str, Path, Path]]:
    if not raw_values:
        return DEFAULT_ROUND_SOURCES
    parsed: list[tuple[str, Path, Path]] = []
    for raw in raw_values:
        try:
            round_name, candidate_path, eval_path = raw.split("::", 2)
        except ValueError as exc:
            raise ValueError(f"Invalid --round-source value: {raw}") from exc
        parsed.append((round_name, Path(candidate_path), Path(eval_path)))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build main and diagnostic benchmark slices from evaluated candidate rounds.")
    parser.add_argument("--out-dir", default="data/hard_hallucination/benchmark_slices_v1")
    parser.add_argument(
        "--round-source",
        action="append",
        default=[],
        help="round_name::candidate_jsonl::eval_jsonl. If omitted, use built-in round0-round3 sources.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    round_sources = parse_round_sources(args.round_source)
    table = build_candidate_table(round_sources)
    main_slice = choose_main_slice(table)
    diagnostic_slice = choose_diagnostic_slice(table)

    write_jsonl(out_dir / "candidate_metrics_table.jsonl", table)
    write_jsonl(out_dir / "main_benchmark_slice.jsonl", main_slice)
    write_jsonl(out_dir / "diagnostic_slice.jsonl", diagnostic_slice)

    summary = {
        "candidate_table_count": len(table),
        "main_slice_count": len(main_slice),
        "diagnostic_slice_count": len(diagnostic_slice),
        "eval_panel_size": EVAL_PANEL_SIZE,
        "sis_threshold": SIS_THRESHOLD,
        # SIS@6/12 命中率：main_slice 中满足条件的候选题比例
        "main_slice_sis_6of12_rate": round(
            sum(1 for r in main_slice if r["sis_6of12"] == 1) / max(len(main_slice), 1), 4
        ),
        "main_slice_by_carrier": dict(
            sorted(
                (
                    (carrier, sum(1 for r in main_slice if r["normalized_answer_carrier"] == carrier))
                    for carrier in {"boolean", "numeric", "entity_set", "citation_set"}
                ),
                key=lambda x: x[0],
            )
        ),
    }
    (out_dir / "slice_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "slice_report.md").write_text(build_report(main_slice, diagnostic_slice, summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
