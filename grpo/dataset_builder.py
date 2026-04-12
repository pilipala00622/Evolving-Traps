#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SYSTEM_PROMPT = (
    "你是一个严格遵守证据边界的企业知识库问答助手。"
    "只能根据给定上下文回答；如果上下文不足以支持确定结论，必须明确说明证据不足，"
    "不要补全、猜测或引用未出现的信息。"
)

SAFE_FALLBACK_SOLUTION = "如果上下文证据不足，应明确说明证据不足，避免补全或猜测。"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def complexity_to_score(bucket: str) -> float:
    mapping = {"easy": 0.20, "medium": 0.55, "hard": 0.85}
    return mapping.get(bucket, 0.50)


def to_json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def render_messages(task: Dict[str, Any]) -> List[Dict[str, str]]:
    user_content = (
        "请仅依据下面上下文回答问题。\n\n"
        f"[Context]\n{task.get('context', '')}\n\n"
        f"[Question]\n{task.get('query', '')}\n\n"
        "要求：如果上下文不足以支持确定答案，请直接说明证据不足。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def render_solution(task: Dict[str, Any]) -> str:
    reference_answer = (task.get("reference_answer") or "").strip()
    if reference_answer:
        return reference_answer

    ground_truth = task.get("ground_truth_final_state") or {}
    answer = str(ground_truth.get("answer") or ground_truth.get("reference_answer") or "").strip()
    if answer:
        return answer

    if ground_truth.get("abstention_required") is True:
        return SAFE_FALLBACK_SOLUTION

    return SAFE_FALLBACK_SOLUTION


def load_hallusea_bundle(
    hallusea_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    tasks = read_jsonl(hallusea_dir / "tasks.jsonl")
    verifiers = read_jsonl(hallusea_dir / "verifiers.jsonl")
    trajectories = read_jsonl(hallusea_dir / "trajectories.jsonl")
    return tasks, verifiers, trajectories


def build_ms_swift_row(
    task: Dict[str, Any],
    verifier: Dict[str, Any],
    trajectory: Dict[str, Any] | None,
) -> Dict[str, Any]:
    messages = render_messages(task)
    metadata = task.get("metadata", {})
    fixed_metrics = metadata.get("fixed_metrics", {})
    difficulty_score = fixed_metrics.get("design_difficulty")
    if difficulty_score is None:
        difficulty_score = complexity_to_score(task.get("complexity_bucket", "medium"))

    verifier_metadata = verifier.get("metadata", {})
    answer_carrier = (
        verifier_metadata.get("answer_carrier")
        or metadata.get("answer_carrier")
        or metadata.get("gene_vector", {}).get("answer_carrier")
        or ""
    )
    reward_spec = {
        "verifier_id": verifier.get("verifier_id", ""),
        "reward_mode": verifier.get("reward_mode", "binary_outcome"),
        "field_rules": verifier.get("field_rules", []),
        "success_criteria": verifier.get("success_criteria", []),
        "failure_reasons": verifier.get("failure_reasons", []),
        "metadata": {
            **verifier_metadata,
            "answer_carrier": answer_carrier,
            "task_id": task.get("task_id", ""),
            "plan_id": task.get("plan_id", ""),
        },
    }
    trajectory = trajectory or {}

    return {
        "messages": messages,
        "solution": render_solution(task),
        "task_id": task.get("task_id", ""),
        "plan_id": task.get("plan_id", ""),
        "verifier_id": verifier.get("verifier_id", ""),
        "target_error_type": task.get("target_error_type", ""),
        "scenario_type": task.get("scenario_type", ""),
        "domain": task.get("domain", ""),
        "complexity_bucket": task.get("complexity_bucket", "medium"),
        "difficulty_score": round(float(difficulty_score), 4),
        "reference_answer": task.get("reference_answer", ""),
        "reference_answer_applicable": bool(metadata.get("reference_answer_applicable", False)),
        "scoring_mode": metadata.get("scoring_mode", "abstention_based"),
        "answer_carrier": answer_carrier,
        "reward_mode": reward_spec["reward_mode"],
        "reward_spec_json": to_json_text(reward_spec),
        "field_rules_json": to_json_text(reward_spec["field_rules"]),
        "success_criteria_json": to_json_text(reward_spec["success_criteria"]),
        "failure_reasons_json": to_json_text(reward_spec["failure_reasons"]),
        "ground_truth_final_state_json": to_json_text(task.get("ground_truth_final_state", {})),
        "task_metadata_json": to_json_text(metadata),
        "bootstrap_trajectory_json": to_json_text(trajectory),
    }


def build_ms_swift_dataset_bundle(hallusea_dir: Path, output_dir: Path) -> Dict[str, Any]:
    tasks, verifiers, trajectories = load_hallusea_bundle(hallusea_dir)
    verifier_by_task = {row["task_id"]: row for row in verifiers}
    trajectory_by_task = {row["task_id"]: row for row in trajectories}

    rows: List[Dict[str, Any]] = []
    for task in tasks:
        task_id = task.get("task_id", "")
        verifier = verifier_by_task.get(task_id)
        if not verifier:
            continue
        rows.append(build_ms_swift_row(task, verifier, trajectory_by_task.get(task_id)))

    rows.sort(key=lambda row: row.get("difficulty_score", 0.5))

    dataset_path = output_dir / "ms_swift_grpo_dataset.jsonl"
    manifest_path = output_dir / "ms_swift_grpo_dataset_manifest.json"
    write_jsonl(dataset_path, rows)

    answer_carrier_distribution: Dict[str, int] = {}
    target_error_distribution: Dict[str, int] = {}
    for row in rows:
        answer_carrier = str(row.get("answer_carrier", "") or "unknown")
        answer_carrier_distribution[answer_carrier] = answer_carrier_distribution.get(answer_carrier, 0) + 1
        target_error = str(row.get("target_error_type", "") or "unknown")
        target_error_distribution[target_error] = target_error_distribution.get(target_error, 0) + 1

    summary = {
        "hallusea_dir": str(hallusea_dir),
        "dataset_path": str(dataset_path),
        "task_count": len(tasks),
        "verifier_count": len(verifiers),
        "trajectory_count": len(trajectories),
        "dataset_count": len(rows),
        "difficulty_distribution": {
            bucket: sum(1 for row in rows if row.get("complexity_bucket") == bucket)
            for bucket in ("easy", "medium", "hard")
        },
        "answer_carrier_distribution": answer_carrier_distribution,
        "target_error_distribution": target_error_distribution,
    }
    write_json(manifest_path, summary)
    return summary


def build_grpo_dataset_bundle(hallusea_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Backward-compatible alias.

    The prepared artifact is now ms-swift native.
    """

    return build_ms_swift_dataset_bundle(hallusea_dir, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ms-swift GRPO dataset bundle from HalluSEA outputs.")
    parser.add_argument("--hallusea-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summary = build_ms_swift_dataset_bundle(Path(args.hallusea_dir), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
