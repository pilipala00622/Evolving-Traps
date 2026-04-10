"""Analyze latent-gene proxy associations from query decisions and sentence annotations.

This script is intentionally lightweight and format-tolerant:
- query decisions can be json / jsonl / csv
- sentence annotations can be json / jsonl / csv

It focuses on three jobs:
1. read query keep/drop (or keep/watch/drop) results
2. read sentence-level annotations
3. compute candidate latent-gene proxy associations with observed failure modes
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple


ASSERTIVE_TOKENS = ["必须", "一定", "就是", "显然", "明确", "直接", "完全", "肯定", "当然", "应当"]
GENERATION_TOKENS = ["写", "续写", "创作", "策划案", "总结", "润色", "改写", "编", "设计", "人物", "剧情"]
BACKGROUND_TOKENS = ["背景", "文档内容", "材料", "介绍", "概要", "根据文档", "根据材料", "参考资料"]
CROSS_CONTEXT_TOKENS = ["结合", "综合", "同时", "分别", "对比", "多个", "所有文档", "全部文档", "跨文档"]
HEDGING_TOKENS = ["可能", "也许", "大概", "或许", "倾向于", "看起来", "似乎"]
DIALOGUE_TOKENS = ["“", "”", "对话", "弟子", "掌门", "下一章", "角色", "剧情"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze latent-gene proxies against failure modes")
    parser.add_argument("--query-decisions", default="", help="keep/drop decision file (json/jsonl/csv)")
    parser.add_argument("--sentence-annotations", required=True, help="sentence annotation file (json/jsonl/csv)")
    parser.add_argument("--output-dir", default="latent_gene_outputs", help="analysis output directory")
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return [data]
        return list(data)
    if suffix == ".jsonl":
        rows = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported file format: {path}")


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def first_present(row: Dict[str, object], candidates: Iterable[str], default=""):
    for key in candidates:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def normalize_query_decisions(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "query_id": str(first_present(row, ["query_id", "item_id", "prompt_id"])),
                "decision": str(first_present(row, ["decision"])),
                "intended_failure_mode": str(
                    first_present(row, ["intended_failure_mode", "confirmed_intended_failure_mode", "target_error_type"])
                ),
                "dominant_observed_failure_mode": str(first_present(row, ["dominant_observed_failure_mode"])),
                "trigger_success_rate": _to_float(first_present(row, ["trigger_success_rate"], 0.0)),
                "query": str(first_present(row, ["query", "question"], "")),
                "reason": str(first_present(row, ["reason", "notes"], "")),
            }
        )
    return [row for row in normalized if row["query_id"]]


def normalize_sentence_annotations(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    normalized = []
    for row in rows:
        annotation_result = row.get("annotation_result", {}) if isinstance(row.get("annotation_result"), dict) else {}
        is_hallucinated = first_present(
            annotation_result or row,
            ["is_hallucinated"],
            None,
        )
        normalized.append(
            {
                "query_id": str(first_present(row, ["query_id", "item_id", "prompt_id"])),
                "response_id": str(first_present(row, ["response_id"], "")),
                "sentence_id": str(first_present(row, ["sentence_id", "annotation_id"], "")),
                "query": str(first_present(row, ["query", "question"], "")),
                "sentence_text": str(first_present(row, ["sentence_text", "sentence"], "")),
                "attribution_type": str(
                    first_present(annotation_result or row, ["attribution_type", "target_error_type", "label"], "")
                ),
                "severity": str(first_present(annotation_result or row, ["severity"], "")),
                "is_hallucinated": truthy(is_hallucinated),
            }
        )
    return [row for row in normalized if row["query_id"] and row["sentence_text"]]


def _to_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def count_hits(text: str, vocab: List[str]) -> int:
    return sum(1 for token in vocab if token in text)


def compute_query_proxies(query_text: str) -> Dict[str, float]:
    text = query_text or ""
    length = len(text)
    return {
        "query_length": float(length),
        "assertiveness_pressure": count_hits(text, ASSERTIVE_TOKENS) / max(1, len(ASSERTIVE_TOKENS)),
        "generation_pressure": count_hits(text, GENERATION_TOKENS) / max(1, len(GENERATION_TOKENS)),
        "background_binding_risk": count_hits(text, BACKGROUND_TOKENS) / max(1, len(BACKGROUND_TOKENS)),
        "cross_context_binding_risk": count_hits(text, CROSS_CONTEXT_TOKENS) / max(1, len(CROSS_CONTEXT_TOKENS)),
    }


def compute_sentence_proxies(sentence_text: str) -> Dict[str, float]:
    text = sentence_text or ""
    return {
        "sentence_length": float(len(text)),
        "assertiveness_density": count_hits(text, ASSERTIVE_TOKENS) / max(1, len(ASSERTIVE_TOKENS)),
        "hedging_density": count_hits(text, HEDGING_TOKENS) / max(1, len(HEDGING_TOKENS)),
        "background_binding_risk": count_hits(text, BACKGROUND_TOKENS) / max(1, len(BACKGROUND_TOKENS)),
        "cross_context_binding_risk": count_hits(text, CROSS_CONTEXT_TOKENS) / max(1, len(CROSS_CONTEXT_TOKENS)),
        "dialogue_narrative_pressure": count_hits(text, DIALOGUE_TOKENS) / max(1, len(DIALOGUE_TOKENS)),
        "numeric_density": sum(ch.isdigit() for ch in text) / max(1, len(text)),
    }


def mean_dict(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: round(mean(row[key] for row in rows), 4) for key in keys}


def build_query_profiles(
    query_rows: List[Dict[str, object]],
    sentence_rows: List[Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    by_query_sentences: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in sentence_rows:
        by_query_sentences[row["query_id"]].append(row)

    query_index = {row["query_id"]: row for row in query_rows}
    query_profiles: Dict[str, Dict[str, object]] = {}
    for query_id, rows in by_query_sentences.items():
        query_text = str(rows[0].get("query", "")) or str(query_index.get(query_id, {}).get("query", ""))
        q_proxies = compute_query_proxies(query_text)
        all_sentence_proxies = [compute_sentence_proxies(row["sentence_text"]) for row in rows]
        hallucinated_rows = [row for row in rows if row["is_hallucinated"]]
        hallucinated_proxies = [compute_sentence_proxies(row["sentence_text"]) for row in hallucinated_rows]
        attr_counts = Counter(row["attribution_type"] for row in hallucinated_rows if row["attribution_type"])
        query_profiles[query_id] = {
            "query_id": query_id,
            "query_text": query_text,
            "decision": query_index.get(query_id, {}).get("decision", ""),
            "intended_failure_mode": query_index.get(query_id, {}).get("intended_failure_mode", ""),
            "dominant_observed_failure_mode": query_index.get(query_id, {}).get("dominant_observed_failure_mode", ""),
            "trigger_success_rate": query_index.get(query_id, {}).get("trigger_success_rate", 0.0),
            "query_proxies": q_proxies,
            "all_sentence_proxy_mean": mean_dict(all_sentence_proxies),
            "hallucinated_sentence_proxy_mean": mean_dict(hallucinated_proxies) if hallucinated_proxies else {},
            "sentence_count": len(rows),
            "hallucinated_sentence_count": len(hallucinated_rows),
            "observed_failure_distribution": dict(attr_counts),
        }
    return query_profiles


def aggregate_proxy_associations(sentence_rows: List[Dict[str, object]]) -> Dict[str, object]:
    by_attr: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    hallucinated_rows = [row for row in sentence_rows if row["is_hallucinated"] and row["attribution_type"]]
    for row in hallucinated_rows:
        by_attr[row["attribution_type"]].append(compute_sentence_proxies(row["sentence_text"]))

    proxy_means_by_attr = {attr: mean_dict(rows) for attr, rows in by_attr.items()}
    attr_counts = {attr: len(rows) for attr, rows in by_attr.items()}
    return {
        "hallucinated_sentence_count": len(hallucinated_rows),
        "failure_mode_counts": attr_counts,
        "proxy_means_by_failure_mode": proxy_means_by_attr,
    }


def aggregate_decision_associations(query_profiles: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    by_decision: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for profile in query_profiles.values():
        decision = profile["decision"] or "unlabeled"
        merged = dict(profile["query_proxies"])
        for key, value in profile["hallucinated_sentence_proxy_mean"].items():
            merged[f"hall_{key}"] = value
        by_decision[decision].append(merged)

    return {
        "query_count_by_decision": {decision: len(rows) for decision, rows in by_decision.items()},
        "proxy_means_by_decision": {
            decision: mean_dict(rows) for decision, rows in by_decision.items() if rows
        },
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, query_profiles: Dict[str, Dict[str, object]], proxy_assoc: Dict[str, object], decision_assoc: Dict[str, object]) -> None:
    lines = [
        "# Latent Gene Analysis",
        "",
        "这份分析尝试用可解释 proxy 去逼近潜在隐性基因，并观察它们与失败模式/筛选决策的关系。",
        "",
        "## Failure-Mode Associations",
        "",
    ]
    for mode, count in sorted(proxy_assoc["failure_mode_counts"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {mode}: {count}")
        lines.append(f"  proxy_means: {proxy_assoc['proxy_means_by_failure_mode'].get(mode, {})}")
    lines.extend(["", "## Decision Associations", ""])
    for decision, count in decision_assoc["query_count_by_decision"].items():
        lines.append(f"- {decision}: {count}")
        lines.append(f"  proxy_means: {decision_assoc['proxy_means_by_decision'].get(decision, {})}")
    lines.extend(["", "## Query Profiles", ""])
    for profile in query_profiles.values():
        lines.append(
            f"- {profile['query_id']} | decision={profile['decision']} | intended={profile['intended_failure_mode']} | observed={profile['dominant_observed_failure_mode']} | trigger_success={profile['trigger_success_rate']}"
        )
        lines.append(f"  query_proxies: {profile['query_proxies']}")
        lines.append(f"  hall_sentence_proxies: {profile['hallucinated_sentence_proxy_mean']}")
        lines.append(f"  observed_failure_distribution: {profile['observed_failure_distribution']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_rows: List[Dict[str, object]] = []
    if args.query_decisions:
        query_rows = normalize_query_decisions(load_records(Path(args.query_decisions)))
    sentence_rows = normalize_sentence_annotations(load_records(Path(args.sentence_annotations)))

    query_profiles = build_query_profiles(query_rows, sentence_rows)
    proxy_assoc = aggregate_proxy_associations(sentence_rows)
    decision_assoc = aggregate_decision_associations(query_profiles)

    write_json(output_dir / "query_profiles.json", query_profiles)
    write_json(output_dir / "proxy_associations.json", proxy_assoc)
    write_json(output_dir / "decision_associations.json", decision_assoc)
    write_markdown(output_dir / "latent_gene_report.md", query_profiles, proxy_assoc, decision_assoc)


if __name__ == "__main__":
    main()
