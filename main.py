"""
EvoHallu - 归因感知的遗传算法驱动幻觉评测题自动生产框架
主入口：演示完整的进化流程

Usage:
    python main.py --mock
    python main.py --seeds seeds.json
    python main.py --seeds seeds.json --generation-model gpt-5.1 --judge-model gpt-5.1
"""

import sys
import os
import json
import argparse

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GAConfig
from core.benchmark_validation import BenchmarkCalibrator, BenchmarkValidator
from core.llm_interface import LLMInterface
from core.evolution import EvolutionEngine


def create_demo_seeds(n: int = 20) -> list:
    """
    创建演示用的种子题配置。
    
    生产环境中，这些配置应从现有150题中提取。
    每个种子包含四层基因的参数值。
    """
    import random

    domains = ["经济金融", "健康医疗", "科技互联网", "教育考试", "法律政务",
               "自然科学", "娱乐休闲", "文化历史"]
    task_types = ["信息定位", "边界感知", "文档整合", "生成控制"]

    seeds = []
    for i in range(n):
        seed = {
            "query": {
                "task_type": task_types[i % len(task_types)],
                "complexity": random.randint(1, 3),
            },
            "context": {
                "domain": domains[i % len(domains)],
                "doc_count": random.randint(1, 8),
                "semantic_similarity": random.uniform(0.3, 0.8),
                "shared_entities": random.randint(0, 4),
                "answer_position": random.choice(["head", "mid", "tail"]),
                "distractor_ratio": random.uniform(0.1, 0.5),
                "total_length": random.choice([10000, 30000, 50000, 80000]),
            },
            "trap": {
                "confusion_pairs": random.randint(0, 3),
                "evidence_clarity": random.uniform(0.4, 0.9),
                "hedging_level": random.randint(0, 2),
                "info_gap": random.uniform(0.1, 0.5),
                "cross_doc_overlap": random.uniform(0.1, 0.5),
            },
            "difficulty": {
                "target_difficulty": random.uniform(0.3, 0.7),
                "step_expansion": random.randint(1, 3),
            },
        }
        seeds.append(seed)

    return seeds


def run_pipeline(
    seeds: list | None = None,
    use_mock: bool = True,
    generation_model: str = "gpt-5.1",
    judge_model: str = "gpt-5.1",
    coarse_filter_model: str = "deepseek-v3.2",
    eval_models: list | None = None,
    config_overrides: dict | None = None,
    build_benchmark: bool = False,
    validation_models: list | None = None,
    anchor_models: list | None = None,
    validation_repeats: int = 2,
):
    """运行基础 GA 管线，可选择 mock 或真实 API。"""
    print("\n" + "=" * 60)
    print("  EvoHallu 基础进化管线")
    print(f"  模式: {'Mock' if use_mock else 'Real API'}")
    print("=" * 60 + "\n")

    # 1. 配置（演示用小规模参数）
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

    # 2. 创建种子题
    seeds = seeds or create_demo_seeds(n=15)
    print(f"种子题数量: {len(seeds)}")

    # 3. 初始化引擎
    llm = LLMInterface(
        use_mock=use_mock,
        generation_model=generation_model,
        judge_model=judge_model,
        coarse_filter_model=coarse_filter_model,
    )
    engine = EvolutionEngine(
        config=config,
        llm=llm,
        seed_questions=seeds,
    )

    # 4. 运行进化
    final_pool = engine.run()

    # 5. 输出结果
    print("\n" + "=" * 60)
    print("  进化结果分析")
    print("=" * 60)

    # 进化报告
    report = engine.get_evolution_report()
    print(f"\n[进化曲线]")
    print(f"  初始最佳fitness: {report['best_fitness_curve'][0]:.4f}")
    print(f"  最终最佳fitness: {report['best_fitness_curve'][-1]:.4f}")
    print(f"  提升幅度: {(report['best_fitness_curve'][-1] - report['best_fitness_curve'][0]) / max(report['best_fitness_curve'][0], 0.001) * 100:.1f}%")

    print(f"\n[归因覆盖度]")
    for attr_type, count in report["attribution_coverage"].items():
        if count > 0:
            print(f"  {attr_type}: {count}题")

    print(f"\n[最终题库]")
    print(f"  题目数量: {len(final_pool)}")

    # 基因分析
    gene_analysis = engine.export_gene_analysis()
    print(f"\n[基因-归因关联分析]（这是EvoLLMs无法产出的洞察）")
    for attr_type, analysis in gene_analysis.items():
        print(f"\n  {attr_type} ({analysis['count']}道题, 平均fitness={analysis['avg_fitness']:.3f}):")
        profile = analysis["gene_profile"]
        # 只显示最关键的参数
        key_params = sorted(
            profile.items(),
            key=lambda x: x[1]["max"] - x[1]["min"],
            reverse=True
        )[:3]
        for param, stats in key_params:
            print(f"    {param}: mean={stats['mean']:.2f} (range: {stats['min']:.2f}~{stats['max']:.2f})")

    # 导出结果
    output = {
        "evolution_report": report,
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
        validator = BenchmarkValidator(
            llm=llm,
            validation_models=validation_models or config.eval_models,
            repeats_per_model=validation_repeats,
        )
        calibrator = BenchmarkCalibrator(
            llm=llm,
            anchor_models=anchor_models or config.eval_models,
        )

        benchmark_items = []
        for ind in final_pool:
            item = validator.validate_candidate(
                ind,
                target_error_type=ind.dominant_attribution_type(),
                scenario_type="static",
                prototype_id=ind.id,
                metadata={
                    "generation": ind.generation,
                    "parent_ids": list(ind.parent_ids),
                },
            )
            item.calibration_stats = calibrator.calibrate_item(item, ind)
            benchmark_items.append(item.to_dict())

        output["benchmark_candidates"] = benchmark_items

    output_path = os.path.join(os.path.dirname(__file__), "evolution_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已导出: {output_path}")

    if build_benchmark:
        benchmark_path = os.path.join(os.path.dirname(__file__), "benchmark_candidates.json")
        with open(benchmark_path, "w", encoding="utf-8") as f:
            json.dump(output["benchmark_candidates"], f, ensure_ascii=False, indent=2, default=str)
        print(f"Benchmark 候选集已导出: {benchmark_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="EvoHallu - 归因感知的遗传算法进化引擎")
    parser.add_argument("--seeds", type=str, help="种子题配置文件路径 (JSON)")
    parser.add_argument("--config", type=str, help="GA配置文件路径 (JSON)")
    parser.add_argument("--mock", action="store_true", help="使用mock LLM模式")
    parser.add_argument("--generation-model", type=str, default="gpt-5.1", help="实例化题目的模型")
    parser.add_argument("--judge-model", type=str, default="gpt-5.1", help="裁判评估模型")
    parser.add_argument("--coarse-filter-model", type=str, default="deepseek-v3.2", help="粗筛模型")
    parser.add_argument(
        "--eval-models",
        nargs="+",
        default=None,
        help="被测模型列表，默认使用配置内的 gpt-5.2 / deepseek-v3.2 / ernie-5.0",
    )
    parser.add_argument("--build-benchmark", action="store_true", help="对最终题库做验证与校准，导出 benchmark 候选集")
    parser.add_argument(
        "--validation-models",
        nargs="+",
        default=None,
        help="验证阶段使用的模型列表，默认复用 eval-models",
    )
    parser.add_argument(
        "--anchor-models",
        nargs="+",
        default=None,
        help="校准阶段使用的锚点模型列表，默认复用 eval-models",
    )
    parser.add_argument(
        "--validation-repeats",
        type=int,
        default=2,
        help="每个验证模型重复评估次数",
    )
    args = parser.parse_args()

    config_overrides = None
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config_overrides = json.load(f)
        print(f"[Config] 已加载配置覆盖: {args.config}")

    if args.seeds:
        with open(args.seeds, "r", encoding="utf-8") as f:
            seeds = json.load(f)
        print(f"[Seed] 已加载外部种子: {args.seeds} ({len(seeds)}条)")
    else:
        seeds = create_demo_seeds(n=15)
        print("[Seed] 未提供外部种子，使用 demo seeds")

    run_pipeline(
        seeds=seeds,
        use_mock=args.mock,
        generation_model=args.generation_model,
        judge_model=args.judge_model,
        coarse_filter_model=args.coarse_filter_model,
        eval_models=args.eval_models,
        config_overrides=config_overrides,
        build_benchmark=args.build_benchmark,
        validation_models=args.validation_models,
        anchor_models=args.anchor_models,
        validation_repeats=args.validation_repeats,
    )


if __name__ == "__main__":
    main()
