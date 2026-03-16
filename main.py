"""
EvoHallu CLI entrypoint.

Pipeline orchestration now lives in `pipelines/evolution_pipeline.py`.
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipelines.evolution_pipeline import run_pipeline


def create_demo_seeds(n: int = 20) -> list:
    """创建演示用的种子题配置。"""
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
    parser.add_argument("--disable-plan-evolution", action="store_true", help="关闭按 plan 分流的进化流程")
    parser.add_argument("--scenario-type", type=str, default="static", help="任务场景类型，如 static / real_time / out_of_date")
    parser.add_argument("--bootstrap-user-model", type=str, default="user-sft-bootstrap", help="导出 TrajectorySpec 时写入的 user model 标识")
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
        plan_evolution=not args.disable_plan_evolution,
        scenario_type=args.scenario_type,
        bootstrap_user_model=args.bootstrap_user_model,
    )


if __name__ == "__main__":
    main()
