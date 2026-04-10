"""
hallusea/curriculum.py
======================
Round-aware HalluSEA 课程管理器。

职责
----
1. 接收当前轮次的 gene population 和 eval_results
2. 按 HALLUSEA_GATES 门槛过滤出可进训练的题
3. Round N>0 额外处理：
   - 加入少量"已解决"基因（防止遗忘），比例由 retention_ratio_solved 控制
   - 只保留当前模型仍然失败（TEHR 仍高）的基因
4. 通过 hallusea.converter.grit_gene_to_benchmark_item() 将基因转为 benchmark_item
5. 通过 core.spec_factory.benchmark_items_to_training_specs() 生成三元组输出
6. 将结果写入 hallusea_dir/：
   - tasks.jsonl
   - verifiers.jsonl
   - trajectories.jsonl
   - pending_human_review.jsonl（需人工确认类型的条目）
   - curriculum_summary.json

依赖
----
  hallusea.converter.grit_gene_to_benchmark_item
  hallusea.converter.filter_archive_for_hallusea
  core.spec_factory.benchmark_items_to_training_specs
  core.training_readiness.filter_verified_release_candidates
  core.round_manager.HALLUSEA_GATES
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.round_manager import HALLUSEA_GATES
from core.spec_factory import benchmark_items_to_training_specs
from core.training_readiness import filter_verified_release_candidates
from hallusea.converter import (
    filter_archive_for_hallusea,
    grit_gene_to_benchmark_item,
)


@dataclass
class HalluSEASignal:
    """curriculum.build() 的返回值，汇总本轮生成的训练信号。"""
    round_id: int
    task_count: int
    eligible_count: int
    solved_retained_count: int
    too_noisy_count: int
    tasks_path: str
    verifiers_path: str
    trajectories_path: str
    pending_review_count: int = 0
    pending_review_path: str = ""
    summary_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HalluSEACurriculum:
    """
    Round-aware 课程管理器。

    Parameters
    ----------
    assistant_model : 生成 TrajectorySpec bootstrap 步骤时使用的模型标识
    user_model      : 模拟用户侧对话的模型标识（bootstrap）
    """

    def __init__(
        self,
        assistant_model: str = "grit_assistant",
        user_model: str = "grit_user_simulator",
    ):
        self.assistant_model = assistant_model
        self.user_model = user_model

    def build(
        self,
        round_id: int,
        population: List[Dict[str, Any]],
        eval_results: List[Dict[str, Any]],
        output_dir: Path,
        prev_tehr_map: Optional[Dict[str, float]] = None,
    ) -> HalluSEASignal:
        """
        生成本轮的 HalluSEA 训练信号。

        Parameters
        ----------
        round_id       : 当前轮次号
        population     : build_gene_population() 输出的种群（含 metrics）
        eval_results   : evaluate_hard_hallucination_candidates 的输出记录列表
        output_dir     : 训练信号写出目录（会自动创建）
        prev_tehr_map  : {gene_id: tehr} 上一轮的 TEHR 快照（用于检测 solved）

        Returns
        -------
        HalluSEASignal
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. 构建 eval_result 索引（gene_id → eval_results）──
        eval_by_gene: Dict[str, List[Dict[str, Any]]] = {}
        for rec in eval_results:
            gid = rec.get("gene_id") or rec.get("seed_id", "")
            if gid:
                eval_by_gene.setdefault(gid, []).append(rec)

        # ── 2. 按 HALLUSEA_GATES 分拣种群 ──────────────────────
        buckets = filter_archive_for_hallusea(population, round_id, prev_tehr_map)
        eligible_genes = buckets["eligible"]
        solved_genes   = buckets["solved"]
        too_noisy      = buckets["too_noisy"]

        # ── 3. 防遗忘：保留少量已解决基因 ───────────────────────
        retention_ratio = HALLUSEA_GATES.get("retention_ratio_solved", 0.20)
        retain_count = max(1, int(len(solved_genes) * retention_ratio))
        # 按上一轮 TEHR 降序排，保留当初最难的那些
        if prev_tehr_map:
            solved_genes_sorted = sorted(
                solved_genes,
                key=lambda g: prev_tehr_map.get(g.get("gene_id", ""), 0.0),
                reverse=True,
            )
        else:
            solved_genes_sorted = solved_genes
        retained_solved = solved_genes_sorted[:retain_count]

        training_genes = eligible_genes + retained_solved

        # ── 4. 转换为 benchmark_item 格式 ───────────────────────
        benchmark_items = [
            grit_gene_to_benchmark_item(gene, eval_by_gene.get(gene.get("gene_id", "")))
            for gene in training_genes
        ]

        # ── 5. filter_verified_release_candidates（二次门控）────
        verified_items = filter_verified_release_candidates(benchmark_items)

        # 找出被过滤掉的（需要人工确认）
        verified_ids = {item["item_id"] for item in verified_items}
        pending_review = [
            item for item in benchmark_items if item["item_id"] not in verified_ids
        ]

        # ── 6. 生成 TaskSpec / VerifierSpec / TrajectorySpec ────
        tasks, verifiers, trajectories = benchmark_items_to_training_specs(
            verified_items,
            assistant_model=self.assistant_model,
            user_model=self.user_model,
        )

        # ── 7. 写出文件 ──────────────────────────────────────────
        tasks_path = output_dir / "tasks.jsonl"
        verifiers_path = output_dir / "verifiers.jsonl"
        trajectories_path = output_dir / "trajectories.jsonl"

        _write_jsonl(tasks_path, [_to_dict(t) for t in tasks])
        _write_jsonl(verifiers_path, [_to_dict(v) for v in verifiers])
        _write_jsonl(trajectories_path, [_to_dict(t) for t in trajectories])

        pending_path = output_dir / "pending_human_review.jsonl"
        if pending_review:
            _write_jsonl(pending_path, pending_review)

        # ── 8. 写出摘要 ──────────────────────────────────────────
        summary = {
            "round_id": round_id,
            "population_size": len(population),
            "eligible_count": len(eligible_genes),
            "solved_count": len(solved_genes),
            "solved_retained_count": len(retained_solved),
            "too_noisy_count": len(too_noisy),
            "verified_count": len(verified_items),
            "pending_review_count": len(pending_review),
            "task_count": len(tasks),
            "hallusea_gates": HALLUSEA_GATES,
        }
        summary_path = output_dir / "curriculum_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[hallusea] round={round_id}: "
              f"{len(eligible_genes)} eligible + {len(retained_solved)} retained_solved "
              f"→ {len(tasks)} tasks written")
        if pending_review:
            print(f"[hallusea] ⚠️  {len(pending_review)} items in pending_human_review.jsonl")

        return HalluSEASignal(
            round_id=round_id,
            task_count=len(tasks),
            eligible_count=len(eligible_genes),
            solved_retained_count=len(retained_solved),
            too_noisy_count=len(too_noisy),
            tasks_path=str(tasks_path),
            verifiers_path=str(verifiers_path),
            trajectories_path=str(trajectories_path),
            pending_review_count=len(pending_review),
            pending_review_path=str(pending_path) if pending_review else "",
            summary_path=str(summary_path),
        )


# ─────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────

def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _to_dict(obj: Any) -> Dict[str, Any]:
    """将 dataclass 或普通 dict 序列化为字典。"""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return dict(obj)
