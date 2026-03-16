"""Top-level evolution and benchmark pipeline orchestration."""

from __future__ import annotations

import json
import os
from typing import Optional

from config import GAConfig
from core.benchmark_validation import BenchmarkCalibrator, BenchmarkValidator
from core.plan_updater import build_updated_plans
from core.plan_workflow import build_evolution_plans, filter_seeds_for_plan, reflect_plan_results
from core.spec_factory import benchmark_items_to_training_specs
from core.llm_interface import LLMInterface
from core.evolution import EvolutionEngine


def run_pipeline(
    *,
    seeds: Optional[list] = None,
    use_mock: bool = True,
    generation_model: str = "gpt-5.1",
    judge_model: str = "gpt-5.1",
    coarse_filter_model: str = "deepseek-v3.2",
    eval_models: Optional[list] = None,
    config_overrides: Optional[dict] = None,
    build_benchmark: bool = False,
    validation_models: Optional[list] = None,
    anchor_models: Optional[list] = None,
    validation_repeats: int = 2,
    plan_evolution: bool = True,
    scenario_type: str = "static",
    bootstrap_user_model: str = "user-sft-bootstrap",
):
    print("\n" + "=" * 60)
    print("  EvoHallu 基础进化管线")
    print(f"  模式: {'Mock' if use_mock else 'Real API'}")
    print("=" * 60 + "\n")

    config_kwargs = dict(
        population_size=100,
        elite_ratio=0.3,
        max_generations=15,
        convergence_patience=5,
        phase1_generations=6,
        target_pool_size=50,
        eval_models=eval_models or ["gpt-5.2", "deepseek-v3.2", "ernie-5.0"],
    )
    if config_overrides:
        config_kwargs.update(config_overrides)
    config = GAConfig(**config_kwargs)

    seeds = seeds or []
    print(f"种子题数量: {len(seeds)}")

    llm = LLMInterface(
        use_mock=use_mock,
        generation_model=generation_model,
        judge_model=judge_model,
        coarse_filter_model=coarse_filter_model,
    )

    plan_reports = []
    plan_reflections = []
    benchmark_items = []

    if plan_evolution:
        plans = build_evolution_plans(seeds, scenario_type=scenario_type)
        print(f"计划流数量: {len(plans)}")
    else:
        plans = []

    if not plans:
        engine = EvolutionEngine(config=config, llm=llm, seed_questions=seeds)
        final_pool = engine.run()
        plan_reports.append(
            {
                "plan_id": "global",
                "seed_count": len(seeds),
                "final_pool_size": len(final_pool),
                "evolution_report": engine.get_evolution_report(),
            }
        )
        global_report = engine.get_evolution_report()
        global_gene_analysis = engine.export_gene_analysis()
    else:
        final_pool = []
        global_report = None
        global_gene_analysis = {}
        validator = None
        calibrator = None

        if build_benchmark:
            validator = BenchmarkValidator(
                llm=llm,
                validation_models=validation_models or config.eval_models,
                repeats_per_model=validation_repeats,
            )
            calibrator = BenchmarkCalibrator(
                llm=llm,
                anchor_models=anchor_models or config.eval_models,
            )

        for plan in plans:
            plan_seeds = filter_seeds_for_plan(seeds, plan)
            if not plan_seeds:
                continue

            print("\n" + "-" * 60)
            print(f"[Plan] {plan.plan_id}")
            print(
                f"  domain={plan.domain} | error_type={plan.error_type} | "
                f"complexity={plan.complexity_bucket} | seeds={len(plan_seeds)}"
            )
            print("-" * 60)

            plan_config_kwargs = dict(config_kwargs)
            plan_config_kwargs["population_size"] = max(20, min(config.population_size, len(plan_seeds) * 8))
            plan_config_kwargs["target_pool_size"] = max(5, min(config.target_pool_size, len(plan_seeds) * 4))
            plan_config = GAConfig(**plan_config_kwargs)

            engine = EvolutionEngine(config=plan_config, llm=llm, seed_questions=plan_seeds)
            plan_final_pool = engine.run()
            final_pool.extend(plan_final_pool)

            plan_gene_analysis = engine.export_gene_analysis()
            plan_report = engine.get_evolution_report()
            plan_reports.append(
                {
                    "plan": plan.to_dict(),
                    "seed_count": len(plan_seeds),
                    "final_pool_size": len(plan_final_pool),
                    "evolution_report": plan_report,
                    "gene_analysis": plan_gene_analysis,
                }
            )
            global_gene_analysis[plan.plan_id] = plan_gene_analysis

            if build_benchmark and validator and calibrator:
                plan_items = []
                for ind in plan_final_pool:
                    item = validator.validate_candidate(
                        ind,
                        target_error_type=plan.error_type or ind.dominant_attribution_type(),
                        scenario_type=plan.scenario_type,
                        prototype_id=ind.id,
                        metadata={
                            "generation": ind.generation,
                            "parent_ids": list(ind.parent_ids),
                            "plan_id": plan.plan_id,
                            "plan_summary": plan.to_dict(),
                            "verification_stage": "verified_data_before_rl",
                        },
                    )
                    item.calibration_stats = calibrator.calibrate_item(item, ind)
                    plan_items.append(item.to_dict())
                benchmark_items.extend(plan_items)
                plan_reflections.append(reflect_plan_results(plan, plan_items))

        report_curves = [
            entry["evolution_report"]["best_fitness_curve"]
            for entry in plan_reports
            if entry.get("evolution_report", {}).get("best_fitness_curve")
        ]
        global_report = {
            "mode": "plan_evolution",
            "plan_count": len(plan_reports),
            "total_final_pool_size": len(final_pool),
            "best_fitness_curve": report_curves[0] if report_curves else [],
            "avg_fitness_curve": [],
            "attribution_coverage": {},
        }

    print("\n" + "=" * 60)
    print("  进化结果分析")
    print("=" * 60)

    report = global_report or {}
    print(f"\n[进化曲线]")
    if report.get("best_fitness_curve"):
        print(f"  初始最佳fitness: {report['best_fitness_curve'][0]:.4f}")
        print(f"  最终最佳fitness: {report['best_fitness_curve'][-1]:.4f}")
        print(
            f"  提升幅度: "
            f"{(report['best_fitness_curve'][-1] - report['best_fitness_curve'][0]) / max(report['best_fitness_curve'][0], 0.001) * 100:.1f}%"
        )
    else:
        print("  当前运行按 plan 分流执行，详见 plan_reports")

    print(f"\n[归因覆盖度]")
    for attr_type, count in report.get("attribution_coverage", {}).items():
        if count > 0:
            print(f"  {attr_type}: {count}题")

    print(f"\n[最终题库]")
    print(f"  题目数量: {len(final_pool)}")

    gene_analysis = global_gene_analysis
    print(f"\n[基因-归因关联分析]（这是EvoLLMs无法产出的洞察）")
    for attr_type, analysis in gene_analysis.items():
        if "count" in analysis:
            print(f"\n  {attr_type} ({analysis['count']}道题, 平均fitness={analysis['avg_fitness']:.3f}):")
            profile = analysis["gene_profile"]
            key_params = sorted(
                profile.items(),
                key=lambda x: x[1]["max"] - x[1]["min"],
                reverse=True
            )[:3]
            for param, stats in key_params:
                print(f"    {param}: mean={stats['mean']:.2f} (range: {stats['min']:.2f}~{stats['max']:.2f})")
        else:
            print(f"\n  Plan {attr_type}: {len(analysis)}个 error-type 子分析")

    updated_plans = build_updated_plans(plans if plan_evolution else [], plan_reflections, iteration=1)

    output = {
        "evolution_report": report,
        "workflow_policy": {
            "verify_before_rl": True,
            "rl_ready_only_after_human_review": True,
            "plan_evolution": plan_evolution,
            "scenario_type": scenario_type,
        },
        "plan_reports": plan_reports,
        "plan_reflections": [reflection.to_dict() for reflection in plan_reflections],
        "updated_plans": [plan.to_dict() for plan in updated_plans],
        "gene_analysis": gene_analysis,
        "final_pool_summary": [
            {
                "id": ind.id,
                "fitness": ind.total_fitness,
                "domain": ind.context_gene.domain,
                "task_type": ind.query_gene.task_type,
                "dominant_attribution": ind.dominant_attribution_type(),
                "gene_vector": ind.get_gene_vector(),
            }
            for ind in final_pool
        ],
    }

    if build_benchmark:
        output["benchmark_candidates"] = benchmark_items
        task_specs, verifier_specs, trajectory_specs = benchmark_items_to_training_specs(
            benchmark_items,
            assistant_model=generation_model,
            user_model=bootstrap_user_model,
        )
        output["task_specs"] = [task.to_dict() for task in task_specs]
        output["verifier_specs"] = [verifier.to_dict() for verifier in verifier_specs]
        output["trajectory_specs"] = [trajectory.to_dict() for trajectory in trajectory_specs]

    output_path = os.path.join(os.getcwd(), "evolution_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已导出: {output_path}")

    if build_benchmark:
        benchmark_path = os.path.join(os.getcwd(), "benchmark_candidates.json")
        with open(benchmark_path, "w", encoding="utf-8") as f:
            json.dump(output["benchmark_candidates"], f, ensure_ascii=False, indent=2, default=str)
        print(f"Benchmark 候选集已导出: {benchmark_path}")

        task_specs_path = os.path.join(os.getcwd(), "task_specs.json")
        with open(task_specs_path, "w", encoding="utf-8") as f:
            json.dump(output["task_specs"], f, ensure_ascii=False, indent=2, default=str)
        print(f"TaskSpec 已导出: {task_specs_path}")

        verifier_specs_path = os.path.join(os.getcwd(), "verifier_specs.json")
        with open(verifier_specs_path, "w", encoding="utf-8") as f:
            json.dump(output["verifier_specs"], f, ensure_ascii=False, indent=2, default=str)
        print(f"VerifierSpec 已导出: {verifier_specs_path}")

        trajectory_specs_path = os.path.join(os.getcwd(), "trajectory_specs.json")
        with open(trajectory_specs_path, "w", encoding="utf-8") as f:
            json.dump(output["trajectory_specs"], f, ensure_ascii=False, indent=2, default=str)
        print(f"TrajectorySpec 已导出: {trajectory_specs_path}")

    updated_plans_path = os.path.join(os.getcwd(), "updated_plans.json")
    with open(updated_plans_path, "w", encoding="utf-8") as f:
        json.dump(output["updated_plans"], f, ensure_ascii=False, indent=2, default=str)
    print(f"Updated plans 已导出: {updated_plans_path}")

    return output
