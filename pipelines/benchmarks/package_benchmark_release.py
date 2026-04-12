#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_release_record(row: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "benchmark_id": row["candidate_id"],
        "split": split,
        "trace_id": row["trace_id"],
        "seed_id": row["seed_id"],
        "source_round": row["source_round"],
        "knowledge_base_category": row["knowledge_base_category"],
        "query": row["query"],
        "target_failure_mode": row["target_error_type"],
        "hard_hallucination_family": row["intended_failure_mechanism"],
        "answer_carrier": row["normalized_answer_carrier"],
        "expected_safe_behavior": row["expected_good_behavior"],
        "verifier_hint": row["verifier_hint"],
        "support_gap_type": row.get("support_gap_type"),
        "verifier_shape": row.get("verifier_shape"),
        "model_labels": row.get("model_labels", {}),
        "metrics": {
            "tehr": row.get("tehr"),
            "purity": row.get("purity"),
            "sis_6of12": row.get("sis_6of12"),
            "selection_score": row.get("selection_score"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Package benchmark slices into a release bundle.")
    parser.add_argument("--base", default="data/hard_hallucination/benchmark_slices_v1")
    parser.add_argument("--out-dir", default="data/hard_hallucination/release_v1")
    parser.add_argument("--release-name", default="hard_hallucination_release_v1")
    args = parser.parse_args()

    base = Path(args.base)
    out_dir = Path(args.out_dir)
    main_rows = read_jsonl(base / "main_benchmark_slice.jsonl")
    diagnostic_rows = read_jsonl(base / "diagnostic_slice.jsonl")

    main_release = [build_release_record(row, "main") for row in main_rows]
    diagnostic_release = [build_release_record(row, "diagnostic") for row in diagnostic_rows]
    full_release = main_release + diagnostic_release

    summary = {
        "release_name": args.release_name,
        "main_count": len(main_release),
        "diagnostic_count": len(diagnostic_release),
        "total_count": len(full_release),
        "main_by_carrier": {
            carrier: sum(1 for row in main_release if row["answer_carrier"] == carrier)
            for carrier in {"boolean", "numeric", "entity_set"}
        },
        "diagnostic_by_carrier": {
            carrier: sum(1 for row in diagnostic_release if row["answer_carrier"] == carrier)
            for carrier in {"boolean", "numeric", "entity_set"}
        },
    }

    write_json(out_dir / "main_release.json", main_release)
    write_json(out_dir / "diagnostic_release.json", diagnostic_release)
    write_json(out_dir / "full_release.json", full_release)
    write_json(out_dir / "release_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
