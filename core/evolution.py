"""
进化主循环：编排整个遗传算法的执行流程

核心流程：
  初始种群 → [评估 → 选择 → 交叉 → 变异 → 评估 → ...]×N代 → 最终筛选

与EvoLLMs的架构差异：
  EvoLLMs: Model1生成 → Model2变异 → Model3评估 → 循环（串行、无种群）
  本框架: 种群并行进化 → 归因感知的多目标优化 → 分阶段策略
"""

import json
import random
from typing import List, Dict
from dataclasses import asdict

from config import GAConfig, ATTRIBUTION_TYPES
from core.gene import GENE_LAYERS, Individual, IndividualFactory
from core.fitness import FitnessEvaluator
from core.operators import Selection, Crossover, Mutation
from core.llm_interface import LLMInterface


class EvolutionEngine:
    """遗传算法进化引擎"""

    def __init__(
        self,
        config: GAConfig = None,
        llm: LLMInterface = None,
        seed_questions: List[Dict] = None,
    ):
        self.config = config or GAConfig()
        self.llm = llm or LLMInterface(use_mock=True)
        self.fitness_eval = FitnessEvaluator(self.config)
        self.mutation_op = Mutation(self.config)
        self.seed_questions = seed_questions or []

        # 进化状态
        self.population: List[Individual] = []
        self.generation: int = 0
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.attribution_coverage_history: List[Dict] = []
        self.hall_of_fame: List[Individual] = []  # 历史最佳个体

    # ===========================================================
    # Phase 0: 初始种群构建
    # ===========================================================

    def initialize_population(self):
        """
        构建初始种群。
        
        策略：用现有150题作为种子，每题生成变体，补充随机个体。
        这比EvoLLMs从零开始效率高得多。
        """
        print(f"[Init] 构建初始种群 (目标: {self.config.population_size})")

        # 1. 从种子题创建变体
        seed_individuals = []
        for seed_config in self.seed_questions:
            parent = IndividualFactory.create_from_seed(seed_config)
            seed_individuals.append(parent)
            # 每个种子题生成若干变体
            n_variants = max(1, self.config.population_size // max(len(self.seed_questions), 1) - 1)
            for _ in range(n_variants):
                variant = IndividualFactory.create_variant(
                    parent,
                    mutation_type=random.choice(["light", "light", "medium"])
                )
                seed_individuals.append(variant)

        # 2. 如果种子不够，用随机个体补充
        n_random = self.config.population_size - len(seed_individuals)
        random_individuals = []
        for _ in range(max(0, n_random)):
            ind = IndividualFactory.create_random()
            random_individuals.append(ind)

        self.population = (seed_individuals + random_individuals)[:self.config.population_size]

        # 3. 实例化所有个体
        for ind in self.population:
            self.llm.instantiate(ind)

        print(f"[Init] 种群构建完成: {len(self.population)}个个体 "
              f"(种子变体: {len(seed_individuals)}, 随机: {len(random_individuals)})")

    # ===========================================================
    # 评估流程：两阶段评估（粗筛→精评）
    # ===========================================================

    def evaluate_population(self):
        """
        评估整个种群的适应度。
        
        两阶段策略：
        1. 粗筛：用1个模型快速过滤无效题（淘汰~60%）
        2. 精评：对通过粗筛的题用3-4个模型完整评估
        """
        print(f"[Gen {self.generation}] 评估种群 ({len(self.population)}个个体)")

        # 更新种群层面的归因分布（用于coverage_bonus计算）
        existing_attrs = [
            ind.model_attributions.get("aggregate", {})
            for ind in self.population if ind.model_attributions
        ]
        self.fitness_eval.update_population_distribution(existing_attrs)

        evaluated = 0
        filtered = 0

        for ind in self.population:
            if ind.total_fitness > 0 and ind.generation < self.generation:
                # 精英个体已有评分，跳过
                continue

            # --- 粗筛阶段 ---
            if not self.llm.coarse_evaluate(ind):
                ind.total_fitness = 0.0
                filtered += 1
                continue

            # --- 精评阶段 ---
            model_rates = {}
            all_attributions = {}

            for model_name in self.config.eval_models:
                halluc_rate, attributions = self.llm.evaluate_with_model(ind, model_name)
                model_rates[model_name] = halluc_rate
                ind.model_attributions[model_name] = attributions

                # 聚合归因
                for attr_type, strength in attributions.items():
                    if attr_type not in all_attributions:
                        all_attributions[attr_type] = 0
                    all_attributions[attr_type] = max(all_attributions[attr_type], strength)

            ind.model_attributions["aggregate"] = all_attributions

            # 有效性评估
            naturalness, is_answerable = self.llm.evaluate_validity(ind)

            # 计算适应度
            result = self.fitness_eval.evaluate(
                model_hallucination_rates=model_rates,
                triggered_attributions=all_attributions,
                has_reference_answer=bool(ind.reference_answer),
                naturalness_score=naturalness,
                is_answerable=is_answerable,
                query_embedding=None,  # 生产环境用真实embedding
                current_generation=self.generation,
            )

            ind.fitness_scores = asdict(result)
            ind.total_fitness = result.total
            evaluated += 1

        print(f"[Gen {self.generation}] 评估完成: 精评{evaluated}, 粗筛淘汰{filtered}")

    # ===========================================================
    # 进化一代
    # ===========================================================

    def evolve_one_generation(self):
        """执行一代进化"""

        # 1. 精英保留
        elites = Selection.elitism(self.population, self.config.elite_ratio)

        # 2. 选择亲本并交叉
        n_offspring = self.config.population_size - len(elites)
        n_pairs = n_offspring // 2 + 1
        parents = Selection.select_parents(self.population, n_pairs)

        offspring = []
        for parent_a, parent_b in parents:
            if len(offspring) >= n_offspring:
                break

            # 交叉始终基于四个显式基因维度：query/context/trap/difficulty
            if random.random() < 0.8:
                child1, child2 = Crossover.uniform_crossover(
                    parent_a, parent_b, self.generation + 1
                )
                offspring.extend([child1, child2])
            else:
                # 归因引导：选择当前种群中最稀缺的归因类型
                dist = self.fitness_eval.population_attribution_dist
                if dist and sum(dist.values()) > 0:
                    # 选最稀缺的类型
                    rarest = min(dist, key=dist.get)
                    child = Crossover.attribution_guided_crossover(
                        parent_a, parent_b, rarest, self.generation + 1
                    )
                    offspring.append(child)
                else:
                    child1, child2 = Crossover.uniform_crossover(
                        parent_a, parent_b, self.generation + 1
                    )
                    offspring.extend([child1, child2])

        # 3. 变异
        mutated_offspring = []
        for child in offspring[:n_offspring]:
            mutated = self.mutation_op.mutate(child)
            mutated_offspring.append(mutated)

        # 4. 实例化新个体
        for ind in mutated_offspring:
            self.llm.instantiate(ind)

        # 5. 组成新种群
        self.population = elites + mutated_offspring
        self.population = self.population[:self.config.population_size]

        self.generation += 1

    # ===========================================================
    # 主进化循环
    # ===========================================================

    def run(self) -> List[Individual]:
        """
        执行完整的进化流程。
        
        Returns:
            最终筛选出的高质量评测题列表
        """
        print("=" * 60)
        print("EvoHallu - 归因感知的遗传算法进化引擎")
        print("=" * 60)

        # Step 1: 初始化
        self.initialize_population()
        self.evaluate_population()
        self._record_stats()

        # Step 2: 进化循环
        for gen in range(self.config.max_generations):
            print(f"\n--- 第 {self.generation + 1} 代 ---")

            self.evolve_one_generation()
            self.evaluate_population()
            self._record_stats()

            # 动态变异率调整
            improved = self._check_improvement()
            self.mutation_op.update_rates(improved)

            # 收敛检测
            if self._check_convergence():
                print(f"[Converge] 连续{self.config.convergence_patience}代无提升，停止进化")
                break

            # 输出进度
            self._print_generation_summary()

        # Step 3: 最终筛选
        final_pool = self._final_selection()

        print(f"\n{'=' * 60}")
        print(f"进化完成！共{self.generation}代")
        print(f"最终题库: {len(final_pool)}道题")
        print(f"API调用: {self.llm.api_call_count}次")
        print(f"{'=' * 60}")

        return final_pool

    # ===========================================================
    # 辅助方法
    # ===========================================================

    def _record_stats(self):
        """记录每代统计信息"""
        fitnesses = [ind.total_fitness for ind in self.population]
        best = max(fitnesses) if fitnesses else 0
        avg = sum(fitnesses) / len(fitnesses) if fitnesses else 0

        self.best_fitness_history.append(best)
        self.avg_fitness_history.append(avg)

        # 归因覆盖度
        self.attribution_coverage_history.append(
            self.fitness_eval.population_attribution_dist.copy()
        )

        # 更新Hall of Fame
        best_ind = max(self.population, key=lambda x: x.total_fitness)
        if not self.hall_of_fame or best_ind.total_fitness > self.hall_of_fame[0].total_fitness:
            self.hall_of_fame.insert(0, best_ind)
            self.hall_of_fame = self.hall_of_fame[:20]  # 保留Top 20

    def _check_improvement(self) -> bool:
        """检查本代是否有fitness提升"""
        if len(self.best_fitness_history) < 2:
            return True
        return self.best_fitness_history[-1] > self.best_fitness_history[-2] * 1.01

    def _check_convergence(self) -> bool:
        """检查是否收敛"""
        if len(self.best_fitness_history) < self.config.convergence_patience:
            return False
        recent = self.best_fitness_history[-self.config.convergence_patience:]
        return max(recent) - min(recent) < 0.01

    def _final_selection(self) -> List[Individual]:
        """
        最终筛选：从进化后的种群中按归因覆盖度分层抽样。
        
        对每种归因类型，选出区分度最高的Top N道题。
        确保最终题库均衡覆盖所有归因类型。
        """
        n_per_type = self.config.target_pool_size // len(ATTRIBUTION_TYPES)

        final = []
        used_ids = set()

        for attr_type in ATTRIBUTION_TYPES:
            # 找出能触发该归因类型的题
            candidates = [
                ind for ind in self.population
                if ind.total_fitness > 0
                   and attr_type in ind.model_attributions.get("aggregate", {})
                   and ind.id not in used_ids
            ]

            # 按区分度排序
            candidates.sort(
                key=lambda x: x.fitness_scores.get("discrimination", 0),
                reverse=True
            )

            selected = candidates[:n_per_type]
            for ind in selected:
                used_ids.add(ind.id)
            final.extend(selected)

        # 如果还没达到目标数量，用剩余最佳题补充
        if len(final) < self.config.target_pool_size:
            remaining = [
                ind for ind in self.population
                if ind.id not in used_ids and ind.total_fitness > 0
            ]
            remaining.sort(key=lambda x: x.total_fitness, reverse=True)
            n_extra = self.config.target_pool_size - len(final)
            final.extend(remaining[:n_extra])

        return final

    def _print_generation_summary(self):
        """打印当代摘要"""
        best = self.best_fitness_history[-1]
        avg = self.avg_fitness_history[-1]
        coverage = self.attribution_coverage_history[-1]

        print(f"  Best Fitness: {best:.4f} | Avg Fitness: {avg:.4f}")
        print(f"  变异率: {self.mutation_op.current_rates}")
        print(f"  归因分布: { {k: v for k, v in coverage.items() if v > 0} }")

    # ===========================================================
    # 分析输出
    # ===========================================================

    def get_evolution_report(self) -> Dict:
        """生成进化过程报告"""
        return {
            "gene_layers": list(GENE_LAYERS),
            "total_generations": self.generation,
            "best_fitness_curve": self.best_fitness_history,
            "avg_fitness_curve": self.avg_fitness_history,
            "final_population_size": len(self.population),
            "attribution_coverage": self.attribution_coverage_history[-1] if self.attribution_coverage_history else {},
            "api_calls": self.llm.api_call_count,
            "hall_of_fame": [
                {
                    "id": ind.id,
                    "fitness": ind.total_fitness,
                    "gene_vector": ind.get_gene_vector(),
                    "dominant_attribution": ind.dominant_attribution_type(),
                }
                for ind in self.hall_of_fame[:10]
            ],
        }

    def export_gene_analysis(self) -> Dict:
        """
        导出基因分析结果 —— 这是EvoLLMs无法产出的洞察。
        
        分析"什么样的基因组合最容易触发什么类型的幻觉"，
        直接为Phase 2的因果实验提供变量选择依据。
        """
        analysis = {}
        for attr_type in ATTRIBUTION_TYPES:
            # 找出该归因类型的高fitness个体
            high_fitness = [
                ind for ind in self.population
                if ind.dominant_attribution_type() == attr_type
                   and ind.total_fitness > 0.5
            ]
            if not high_fitness:
                continue

            # 统计这些个体的基因参数分布
            gene_vectors = [ind.get_gene_vector() for ind in high_fitness]
            numeric_keys = [
                "complexity", "doc_count", "semantic_similarity", "shared_entities",
                "distractor_ratio", "confusion_pairs", "evidence_clarity",
                "hedging_level", "info_gap", "cross_doc_overlap",
            ]
            gene_stats = {}
            for key in numeric_keys:
                values = [g[key] for g in gene_vectors if isinstance(g.get(key), (int, float))]
                if values:
                    gene_stats[key] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                    }

            analysis[attr_type] = {
                "count": len(high_fitness),
                "avg_fitness": sum(i.total_fitness for i in high_fitness) / len(high_fitness),
                "gene_profile": gene_stats,
            }

        return analysis
