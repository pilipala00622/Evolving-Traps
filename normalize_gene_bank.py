#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


FAILURE_MECHANISM_MAP = {
    "general_to_special_population_extrapolation_under_weak_evidence": "weak_evidence_to_strong_conclusion",
    "fills missing ASP/cost/margin/tax inputs to satisfy an exact-profit request": "missing_info_hard_answer",
}


def normalize_target_error_type(value: Any, source_target_error_type: Any) -> Any:
    if not isinstance(value, str):
        return source_target_error_type or value
    normalized = value.strip().lower()
    raw = value.strip()
    mapping = {
        "overreach_inference": "越权推理",
        "unsupported_fabrication": "无中生有",
        "unsupported_numeric_fabrication": "无中生有",
        "unsupported_slot_fabrication": "生成错误",
        "unsupported_case_qualification": "生成错误",
        "边界误判错误": "越权推理",
        "generation_error": "生成错误",
        "越权推理": "越权推理",
        "无中生有": "无中生有",
        "生成错误": "生成错误",
    }
    if normalized in mapping:
        return mapping[normalized]
    if "越权" in raw:
        return "越权推理"
    if "无中生有" in raw or "编造" in raw:
        return "无中生有"
    if "生成错误" in raw or "错误生成" in raw:
        return "生成错误"
    return source_target_error_type or value


def normalize_failure_mechanism(value: Any, mechanism_hint: Any) -> Any:
    if isinstance(value, str) and value in FAILURE_MECHANISM_MAP:
        return FAILURE_MECHANISM_MAP[value]
    if mechanism_hint == "弱证据强结论":
        return "weak_evidence_to_strong_conclusion"
    if mechanism_hint == "缺失关键信息下硬答":
        return "missing_info_hard_answer"
    if mechanism_hint == "背景证据当直接证据":
        return "background_as_direct_evidence"
    return value


def normalize_support_gap_type(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return value.replace(" ", "")


def normalize_answer_carrier(value: Any, source_answer_carrier: Any) -> Any:
    if not isinstance(value, str):
        return source_answer_carrier or value
    raw = value.strip()
    lowered = raw.lower()
    if lowered in {"numeric", "numeric_value"} or "numeric" in lowered or "数值" in raw:
        return "numeric"
    if lowered in {"boolean", "bool"} or "boolean" in lowered or "布尔" in raw or "是非" in raw:
        return "boolean"
    if lowered in {"entity_set"} or "entity" in lowered or "实体" in raw:
        return "entity_set"
    if lowered in {"citation_set"} or "citation" in lowered or "引用" in raw or "出处" in raw:
        return "citation_set"
    return source_answer_carrier or value


def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    if "error" in out:
        return out
    out["target_error_type"] = normalize_target_error_type(
        out.get("target_error_type"), out.get("source_target_error_type")
    )
    out["failure_mechanism"] = normalize_failure_mechanism(out.get("failure_mechanism"), out.get("mechanism"))
    out["support_gap_type"] = normalize_support_gap_type(out.get("support_gap_type"))
    out["answer_carrier"] = normalize_answer_carrier(out.get("answer_carrier"), out.get("source_answer_carrier"))
    return out


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize extracted gene bank fields.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    normalized = [normalize_record(rec) for rec in records]
    write_jsonl(Path(args.output), normalized)


if __name__ == "__main__":
    main()
