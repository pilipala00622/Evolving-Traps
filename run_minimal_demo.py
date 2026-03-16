"""Run a minimal closed-loop demo on the phase-1 evaluation data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple

from config import ATTRIBUTION_TYPES
from pipelines.evolution_pipeline import run_pipeline


DEFAULT_SOURCE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "第一阶段-14个模型-gpt51评测",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "demo_outputs",
    "minimal_closed_loop_demo",
)

TARGET_MODELS = ["gpt-5.2", "deepseek-v3.2", "ernie-5.0"]
MODEL_SKILL = {
    "gpt-5.2": 0.78,
    "deepseek-v3.2": 0.62,
    "ernie-5.0": 0.46,
}
TARGET_TAXONOMY = {
    "缺证断言": {
        "error_location": "final_claim_error",
        "error_mechanism": "unsupported_inference",
        "impact_level": "partial_misleading",
    },
    "引入新事实": {
        "error_location": "final_claim_error",
        "error_mechanism": "fabricated_detail",
        "impact_level": "partial_misleading",
    },
    "引用错误": {
        "error_location": "evidence_alignment_error",
        "error_mechanism": "conflicting_evidence_ignored",
        "impact_level": "partial_misleading",
    },
    "错误拼接": {
        "error_location": "reasoning_error",
        "error_mechanism": "unsupported_inference",
        "impact_level": "decision_blocking",
    },
}


@contextmanager
def pushd(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal closed-loop demo.")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR, help="Phase-1 data directory")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Demo output directory")
    parser.add_argument("--max-seeds", type=int, default=18, help="Max number of seeds to use")
    parser.add_argument("--population-size", type=int, default=24, help="Demo GA population size")
    parser.add_argument("--max-generations", type=int, default=4, help="Demo GA generations")
    parser.add_argument("--validation-repeats", type=int, default=2, help="Validation repeats per model")
    parser.add_argument("--rollout-repeats", type=int, default=5, help="Rollout repeats per task/model")
    parser.add_argument("--subset-size", type=int, default=12, help="Benchmark subset size after GA selection")
    return parser.parse_args()


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_seed(*parts: object) -> int:
    raw = "::".join(str(part) for part in parts)
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16)


def load_demo_seeds(source_dir: str, max_seeds: int) -> List[Dict]:
    seed_path = os.path.join(source_dir, "seed_questions.json")
    seeds = read_json(seed_path)

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for seed in seeds:
        grouped[seed["trap"]["target_attribution"]].append(seed)

    selected: List[Dict] = []
    while len(selected) < min(max_seeds, len(seeds)):
        progressed = False
        for attr in sorted(grouped):
            if grouped[attr] and len(selected) < max_seeds:
                selected.append(grouped[attr].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def build_taxonomy_v1() -> Dict[str, object]:
    labels = {}
    for label, meta in ATTRIBUTION_TYPES.items():
        fallback = TARGET_TAXONOMY.get(
            label,
            {
                "error_location": "reasoning_error",
                "error_mechanism": "unsupported_inference",
                "impact_level": "partial_misleading",
            },
        )
        labels[label] = {
            "definition": meta["description"],
            "error_location": fallback["error_location"],
            "error_mechanism": fallback["error_mechanism"],
            "impact_level": fallback["impact_level"],
            "trigger_genes": meta["trigger_genes"],
            "positive_examples": [],
            "negative_examples": [],
            "decision_rule": f"当主要失败模式符合“{meta['description']}”时标注为 `{label}`。",
        }
    labels["unknown_or_new_pattern"] = {
        "definition": "当前 taxonomy 无法稳定解释的新失败模式。",
        "error_location": "unknown",
        "error_mechanism": "unknown",
        "impact_level": "unknown",
        "trigger_genes": [],
        "positive_examples": [],
        "negative_examples": [],
        "decision_rule": "自动归因低置信、多人分歧大或 verifier 边界异常时进入该桶。",
    }
    return {
        "version": "taxonomy_v1_demo",
        "layers": ["error_location", "error_mechanism", "impact_level"],
        "labels": labels,
    }


def write_taxonomy_markdown(path: str, taxonomy: Dict[str, object]) -> None:
    lines = [
        "# Taxonomy v1 Demo",
        "",
        "这是最小 demo 使用的分层归因体系。",
        "",
    ]
    for label, spec in taxonomy["labels"].items():
        lines.extend(
            [
                f"## {label}",
                "",
                f"- definition: {spec['definition']}",
                f"- error_location: {spec['error_location']}",
                f"- error_mechanism: {spec['error_mechanism']}",
                f"- impact_level: {spec['impact_level']}",
                f"- decision_rule: {spec['decision_rule']}",
                "",
            ]
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def score_review_candidate(item: Dict[str, object]) -> float:
    validation = item["validation_stats"]
    calibration = item["calibration_stats"]
    trigger = validation.get("target_error_trigger_rate", 0.0)
    leakage = validation.get("non_target_error_leakage", 1.0)
    naturalness = validation.get("naturalness_mean", 0.0) / 5.0
    score_mean = calibration.get("score_mean", 50.0) / 100.0
    recommended = 0.15 if calibration.get("recommended_for_release") else 0.0
    return round(
        0.32 * trigger
        + 0.22 * (1.0 - leakage)
        + 0.18 * naturalness
        + 0.18 * score_mean
        + 0.10 * recommended,
        4,
    )


def build_demo_review(benchmark_candidates: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    candidates = copy.deepcopy(benchmark_candidates)
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for item in candidates:
        grouped[item["target_error_type"]].append(item)

    approved_ids = set()
    for _, items in grouped.items():
        ranked = sorted(items, key=score_review_candidate, reverse=True)
        keep = max(2, math.ceil(len(ranked) * 0.35))
        approved_ids.update(item["item_id"] for item in ranked[:keep])

    review_rows = []
    for item in candidates:
        score = score_review_candidate(item)
        approved = item["item_id"] in approved_ids
        review_result = {
            "reviewer": "demo-auto-reviewer",
            "decision": "approve" if approved else "revise",
            "confirmed_target_error_type": item["target_error_type"],
            "confirmed_scenario_type": item["scenario_type"],
            "reference_answer_supported": True,
            "final_state_is_correctly_specified": True,
            "verifier_design_is_feasible": True,
            "reward_should_be_verifiable": True,
            "query_is_natural": item["validation_stats"].get("naturalness_mean", 0.0) >= 3.5,
            "time_metadata_correct": item["scenario_type"] != "static",
            "is_single_target_error": item["validation_stats"].get("non_target_error_leakage", 1.0) <= 0.95,
            "release_priority": "high" if approved else "medium",
            "issue_tags": [] if approved else ["needs_manual_calibration"],
            "notes": f"demo review score={score}",
        }
        item["human_review"]["status"] = "approved" if approved else "needs_revision"
        item["human_review"]["reviewer"] = review_result["reviewer"]
        item["human_review"]["notes"] = review_result["notes"]
        item["human_review"]["review_result"] = review_result

        review_rows.append(
            {
                "item_id": item["item_id"],
                "target_error_type": item["target_error_type"],
                "review_result": review_result,
            }
        )

    approved_items = [item for item in candidates if item["human_review"]["status"] == "approved"]
    return candidates, approved_items + review_rows


def index_by_task_id(rows: List[Dict[str, object]], key: str) -> Dict[str, Dict[str, object]]:
    return {row[key]: row for row in rows}


def verifier_matches(final_state: Dict[str, object], verifier_spec: Dict[str, object]) -> Tuple[bool, str, List[str]]:
    mismatches = []
    for rule in verifier_spec.get("field_rules", []):
        field = rule["field"]
        expected = rule["expected_value"]
        if rule.get("required", False) and field not in final_state:
            mismatches.append(field)
            continue
        if final_state.get(field) != expected:
            mismatches.append(field)
    if mismatches:
        return False, "final_state_mismatch", mismatches
    return True, "", []


def rollout_success_probability(item: Dict[str, object], model_name: str) -> float:
    validation = item["validation_stats"]
    calibration = item["calibration_stats"]
    skill = MODEL_SKILL[model_name]
    hardness = validation.get("target_error_trigger_rate", 0.0)
    leakage = validation.get("non_target_error_leakage", 1.0)
    naturalness = validation.get("naturalness_mean", 3.5)
    score_mean = calibration.get("score_mean", 50.0)
    difficulty_bucket = calibration.get("difficulty_bucket", "medium")

    bucket_penalty = {
        "hard": 0.14,
        "medium": 0.08,
        "easy": 0.02,
    }.get(difficulty_bucket, 0.08)
    support_bonus = (naturalness - 3.0) * 0.04 + max(0.0, score_mean - 55.0) / 200.0
    prob = skill + support_bonus - 0.25 * hardness - 0.10 * leakage - bucket_penalty
    return max(0.05, min(0.95, prob))


def build_final_state(task_spec: Dict[str, object], success: bool, rng: random.Random) -> Dict[str, object]:
    state = copy.deepcopy(task_spec["ground_truth_final_state"])
    if success:
        return state

    corrupted = copy.deepcopy(state)
    mutation_type = rng.choice(["reference_answer", "target_error_type", "drop_field"])
    if mutation_type == "reference_answer":
        corrupted["reference_answer"] = f"{corrupted['reference_answer']} [demo mismatch]"
    elif mutation_type == "target_error_type":
        corrupted["target_error_type"] = "unknown_or_new_pattern"
    else:
        corrupted.pop("reference_answer", None)
    return corrupted


def predicted_attribution(item: Dict[str, object], success: bool, unstable: bool) -> Dict[str, str]:
    if unstable:
        return {
            "error_location": "unknown",
            "error_mechanism": "unknown",
            "impact_level": "unknown",
            "taxonomy_label": "unknown_or_new_pattern",
        }
    label = item["target_error_type"] if not success else "none"
    mapping = TARGET_TAXONOMY.get(
        item["target_error_type"],
        {
            "error_location": "reasoning_error",
            "error_mechanism": "unsupported_inference",
            "impact_level": "partial_misleading",
        },
    )
    return {
        "error_location": mapping["error_location"] if not success else "none",
        "error_mechanism": mapping["error_mechanism"] if not success else "none",
        "impact_level": mapping["impact_level"] if not success else "none",
        "taxonomy_label": label,
    }


def run_rollout_demo(
    approved_items: List[Dict[str, object]],
    task_specs: List[Dict[str, object]],
    verifier_specs: List[Dict[str, object]],
    rollout_repeats: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    task_by_id = index_by_task_id(task_specs, "task_id")
    verifier_by_task = {row["task_id"]: row for row in verifier_specs}
    item_by_task = {row["item_id"]: row for row in approved_items}

    rows = []
    task_stats = defaultdict(lambda: {"rewards": [], "by_model": defaultdict(list), "failures": Counter()})

    for item in approved_items:
        task_id = item["item_id"]
        task_spec = task_by_id[task_id]
        verifier_spec = verifier_by_task[task_id]

        for model_name in TARGET_MODELS:
            prob = rollout_success_probability(item, model_name)
            for run_id in range(rollout_repeats):
                rng = random.Random(stable_seed(task_id, model_name, run_id))
                success = rng.random() < prob
                final_state = build_final_state(task_spec, success, rng)
                passed, failure_reason, mismatched_fields = verifier_matches(final_state, verifier_spec)
                reward = 1.0 if passed else 0.0
                unstable = (0.35 < prob < 0.65) and not passed
                attr = predicted_attribution(item, passed, unstable)
                latency_ms = 600 + int(rng.random() * 700)

                row = {
                    "task_id": task_id,
                    "model_name": model_name,
                    "run_id": run_id,
                    "reward": reward,
                    "success_probability": round(prob, 4),
                    "final_state": final_state,
                    "failure_reason": failure_reason,
                    "mismatched_fields": mismatched_fields,
                    "predicted_attribution": attr,
                    "latency_ms": latency_ms,
                }
                rows.append(row)
                task_stats[task_id]["rewards"].append(reward)
                task_stats[task_id]["by_model"][model_name].append(reward)
                if failure_reason:
                    task_stats[task_id]["failures"][failure_reason] += 1

    per_task = {}
    overall_by_model = defaultdict(list)
    for task_id, stats in task_stats.items():
        per_model_mean = {
            model_name: round(mean(values), 4)
            for model_name, values in stats["by_model"].items()
        }
        for model_name, values in stats["by_model"].items():
            overall_by_model[model_name].extend(values)
        per_task[task_id] = {
            "reward_mean": round(mean(stats["rewards"]), 4),
            "reward_std": round(pstdev(stats["rewards"]) if len(stats["rewards"]) > 1 else 0.0, 4),
            "per_model_mean": per_model_mean,
            "top_failure_reason": stats["failures"].most_common(1)[0][0] if stats["failures"] else "",
        }

    overall = {
        "model_reward_mean": {
            model_name: round(mean(values), 4) if values else 0.0
            for model_name, values in overall_by_model.items()
        },
        "model_reward_std": {
            model_name: round(pstdev(values), 4) if len(values) > 1 else 0.0
            for model_name, values in overall_by_model.items()
        },
        "task_count": len(per_task),
        "rollout_count": len(rows),
    }
    return rows, {"per_task": per_task, "overall": overall}


def select_benchmark_subset_with_ga(
    approved_items: List[Dict[str, object]],
    rollout_summary: Dict[str, object],
    subset_size: int,
) -> Dict[str, object]:
    task_metrics = rollout_summary["per_task"]
    task_ids = [item["item_id"] for item in approved_items if item["item_id"] in task_metrics]
    subset_size = min(subset_size, len(task_ids))
    if subset_size == 0:
        return {"selected_task_ids": [], "best_fitness": 0.0, "history": []}

    meta_by_task = {item["item_id"]: item for item in approved_items}

    def subset_fitness(task_subset: List[str]) -> Tuple[float, Dict[str, float]]:
        domains = Counter(meta_by_task[task_id]["metadata"]["plan_summary"]["domain"] for task_id in task_subset)
        attrs = Counter(meta_by_task[task_id]["target_error_type"] for task_id in task_subset)
        reward_stds = [task_metrics[task_id]["reward_std"] for task_id in task_subset]
        separations = []
        for task_id in task_subset:
            per_model = task_metrics[task_id]["per_model_mean"]
            if all(model in per_model for model in TARGET_MODELS):
                separations.append(per_model["gpt-5.2"] - per_model["ernie-5.0"])
        coverage = 0.5 * (len(domains) / max(1, len({meta_by_task[t]["metadata"]["plan_summary"]["domain"] for t in task_ids})))
        coverage += 0.5 * (len(attrs) / max(1, len({meta_by_task[t]["target_error_type"] for t in task_ids})))
        stability = 1.0 - min(1.0, mean(reward_stds) if reward_stds else 1.0)
        separability = max(0.0, min(1.0, mean(separations) if separations else 0.0))
        agreement = mean(
            1.0 if meta_by_task[task_id]["human_review"]["review_result"]["is_single_target_error"] else 0.7
            for task_id in task_subset
        )
        fitness = 0.3 * coverage + 0.35 * stability + 0.25 * separability + 0.1 * agreement
        return round(fitness, 4), {
            "coverage": round(coverage, 4),
            "stability": round(stability, 4),
            "separability": round(separability, 4),
            "agreement": round(agreement, 4),
        }

    population = []
    rng = random.Random(42)
    for _ in range(24):
        population.append(sorted(rng.sample(task_ids, subset_size)))

    history = []
    best_subset = population[0]
    best_score, best_breakdown = subset_fitness(best_subset)

    for generation in range(8):
        scored = []
        for subset in population:
            score, breakdown = subset_fitness(subset)
            scored.append((score, subset, breakdown))
        scored.sort(key=lambda row: row[0], reverse=True)
        history.append(
            {
                "generation": generation,
                "best_fitness": scored[0][0],
                "avg_fitness": round(mean(row[0] for row in scored), 4),
            }
        )
        if scored[0][0] > best_score:
            best_score, best_subset, best_breakdown = scored[0]

        elites = [row[1] for row in scored[:6]]
        next_population = elites[:]
        while len(next_population) < len(population):
            parent_a, parent_b = rng.sample(elites, 2)
            cut = rng.randint(1, subset_size - 1) if subset_size > 1 else 1
            merged = list(dict.fromkeys(parent_a[:cut] + parent_b[cut:]))
            while len(merged) < subset_size:
                candidate = rng.choice(task_ids)
                if candidate not in merged:
                    merged.append(candidate)
            if len(merged) > subset_size:
                merged = merged[:subset_size]
            if rng.random() < 0.35:
                replace_idx = rng.randrange(subset_size)
                replacement = rng.choice(task_ids)
                if replacement not in merged:
                    merged[replace_idx] = replacement
            next_population.append(sorted(merged))
        population = next_population[: len(population)]

    return {
        "selected_task_ids": best_subset,
        "best_fitness": best_score,
        "fitness_breakdown": best_breakdown,
        "history": history,
    }


def analyze_failure_patterns(
    rollout_rows: List[Dict[str, object]],
    approved_items: List[Dict[str, object]],
) -> Dict[str, object]:
    item_by_task = {item["item_id"]: item for item in approved_items}
    buckets = defaultdict(list)
    for row in rollout_rows:
        if row["reward"] == 1.0:
            continue
        task = item_by_task[row["task_id"]]
        prob = row["success_probability"]
        if row["predicted_attribution"]["taxonomy_label"] == "unknown_or_new_pattern":
            buckets["borderline_verifier_or_ambiguous_task"].append(row)
        elif task["target_error_type"] == "缺证断言":
            buckets["evidence_gap_not_explicitly_modeled"].append(row)
        elif task["target_error_type"] == "引入新事实":
            buckets["fact_fabrication_under_sparse_context"].append(row)
        elif prob < 0.35:
            buckets["high_difficulty_cascade"].append(row)

    suggestions = []
    for label, rows in sorted(buckets.items(), key=lambda item: len(item[1]), reverse=True):
        suggestions.append(
            {
                "candidate_label": label,
                "count": len(rows),
                "promote_to_taxonomy_v2": len(rows) >= 3,
                "example_task_ids": sorted({row["task_id"] for row in rows})[:5],
            }
        )
    return {
        "unknown_pattern_count": sum(len(rows) for rows in buckets.values()),
        "suggestions": suggestions,
    }


def write_taxonomy_v2_markdown(path: str, taxonomy_v1: Dict[str, object], failure_analysis: Dict[str, object]) -> None:
    lines = [
        "# Taxonomy v2 Demo Suggestions",
        "",
        "这一版不是正式 taxonomy，只展示从失败样本回流出的新增候选模式。",
        "",
    ]
    promoted = [item for item in failure_analysis["suggestions"] if item["promote_to_taxonomy_v2"]]
    if not promoted:
        lines.append("- 本轮 demo 没有足够高频的新模式。")
    else:
        for item in promoted:
            lines.extend(
                [
                    f"## {item['candidate_label']}",
                    "",
                    f"- count: {item['count']}",
                    f"- example_task_ids: {', '.join(item['example_task_ids'])}",
                    "- suggested_layer_mapping: error_location=reasoning_error, error_mechanism=unsupported_inference, impact_level=partial_misleading",
                    "",
                ]
            )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_demo_report(
    output_path: str,
    taxonomy_path: str,
    seed_count: int,
    benchmark_candidates: List[Dict[str, object]],
    approved_items: List[Dict[str, object]],
    rollout_summary: Dict[str, object],
    subset_result: Dict[str, object],
    failure_analysis: Dict[str, object],
) -> None:
    attr_counter = Counter(item["target_error_type"] for item in benchmark_candidates)
    approved_counter = Counter(item["target_error_type"] for item in approved_items)
    lines = [
        "# Minimal Closed-Loop Demo Report",
        "",
        f"- taxonomy_v1: `{taxonomy_path}`",
        f"- seed_count: {seed_count}",
        f"- benchmark_candidate_count: {len(benchmark_candidates)}",
        f"- approved_count: {len(approved_items)}",
        f"- rollout_count: {rollout_summary['overall']['rollout_count']}",
        "",
        "## Candidate Distribution",
        "",
    ]
    for label, count in sorted(attr_counter.items()):
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Approved Distribution", ""])
    for label, count in sorted(approved_counter.items()):
        lines.append(f"- {label}: {count}")
    lines.extend(
        [
            "",
            "## Rollout Summary",
            "",
        ]
    )
    for model_name, value in rollout_summary["overall"]["model_reward_mean"].items():
        std = rollout_summary["overall"]["model_reward_std"][model_name]
        lines.append(f"- {model_name}: mean_reward={value}, std={std}")
    lines.extend(
        [
            "",
            "## GA Subset",
            "",
            f"- selected_task_count: {len(subset_result['selected_task_ids'])}",
            f"- best_fitness: {subset_result['best_fitness']}",
            f"- fitness_breakdown: {subset_result['fitness_breakdown']}",
            "",
            "## Taxonomy v2 Candidates",
            "",
            f"- unknown_pattern_count: {failure_analysis['unknown_pattern_count']}",
        ]
    )
    for item in failure_analysis["suggestions"]:
        lines.append(
            f"- {item['candidate_label']}: count={item['count']}, promote={item['promote_to_taxonomy_v2']}"
        )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    taxonomy_v1 = build_taxonomy_v1()
    taxonomy_v1_json = os.path.join(args.output_dir, "taxonomy_v1_demo.json")
    taxonomy_v1_md = os.path.join(args.output_dir, "taxonomy_v1_demo.md")
    write_json(taxonomy_v1_json, taxonomy_v1)
    write_taxonomy_markdown(taxonomy_v1_md, taxonomy_v1)

    seeds = load_demo_seeds(args.source_dir, args.max_seeds)
    seed_output_path = os.path.join(args.output_dir, "demo_seed_questions.json")
    write_json(seed_output_path, seeds)

    config_overrides = {
        "population_size": args.population_size,
        "max_generations": args.max_generations,
        "target_pool_size": max(8, args.population_size // 2),
        "phase1_generations": max(2, min(3, args.max_generations)),
        "convergence_patience": 3,
    }

    with pushd(args.output_dir):
        run_pipeline(
            seeds=seeds,
            use_mock=True,
            generation_model="gpt-5.1",
            judge_model="gpt-5.1",
            coarse_filter_model="deepseek-v3.2",
            eval_models=TARGET_MODELS,
            config_overrides=config_overrides,
            build_benchmark=True,
            validation_models=TARGET_MODELS,
            anchor_models=TARGET_MODELS,
            validation_repeats=args.validation_repeats,
            plan_evolution=True,
            scenario_type="static",
            bootstrap_user_model="demo-user-sim",
        )

    benchmark_candidates = read_json(os.path.join(args.output_dir, "benchmark_candidates.json"))
    task_specs = read_json(os.path.join(args.output_dir, "task_specs.json"))
    verifier_specs = read_json(os.path.join(args.output_dir, "verifier_specs.json"))

    reviewed_candidates, approved_plus_reviews = build_demo_review(benchmark_candidates)
    approved_items = [item for item in reviewed_candidates if item["human_review"]["status"] == "approved"]
    review_rows = [row for row in approved_plus_reviews if "review_result" in row and "item_id" in row and "target_error_type" in row]

    write_json(os.path.join(args.output_dir, "benchmark_candidates.reviewed.demo.json"), reviewed_candidates)
    write_json(os.path.join(args.output_dir, "benchmark_release_candidates.demo.json"), approved_items)
    write_jsonl(os.path.join(args.output_dir, "benchmark_review_tasks.demo_reviewed.jsonl"), review_rows)

    rollout_rows, rollout_summary = run_rollout_demo(
        approved_items=approved_items,
        task_specs=task_specs,
        verifier_specs=verifier_specs,
        rollout_repeats=args.rollout_repeats,
    )
    write_jsonl(os.path.join(args.output_dir, "rollout_results.demo.jsonl"), rollout_rows)
    write_json(os.path.join(args.output_dir, "rollout_summary.demo.json"), rollout_summary)

    subset_result = select_benchmark_subset_with_ga(
        approved_items=approved_items,
        rollout_summary=rollout_summary,
        subset_size=args.subset_size,
    )
    write_json(os.path.join(args.output_dir, "benchmark_subset_v1.demo.json"), subset_result)

    failure_analysis = analyze_failure_patterns(rollout_rows, approved_items)
    write_json(os.path.join(args.output_dir, "failure_pattern_analysis.demo.json"), failure_analysis)
    write_taxonomy_v2_markdown(
        os.path.join(args.output_dir, "taxonomy_v2_demo.md"),
        taxonomy_v1,
        failure_analysis,
    )

    build_demo_report(
        output_path=os.path.join(args.output_dir, "demo_report.md"),
        taxonomy_path=taxonomy_v1_md,
        seed_count=len(seeds),
        benchmark_candidates=benchmark_candidates,
        approved_items=approved_items,
        rollout_summary=rollout_summary,
        subset_result=subset_result,
        failure_analysis=failure_analysis,
    )

    print(f"Demo finished. Outputs are in: {args.output_dir}")


if __name__ == "__main__":
    main()
