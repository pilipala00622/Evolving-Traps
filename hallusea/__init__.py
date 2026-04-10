"""
hallusea/
=========
HalluSEA 训练信号生成包。

将 GRIT 进化产出的高适应度基因档案转换为可供 RLVR 训练使用的三元组：
  TaskSpec + VerifierSpec + TrajectorySpec

子模块：
  converter   : GRIT eval_result / gene 格式 → benchmark_item 中间格式
  curriculum  : Round-aware 课程管理（决定哪些题进本轮训练、哪些保留防遗忘）
  round_state : 轮次状态持久化（对 core.round_manager.RoundState 的便捷封装）
"""

from hallusea.curriculum import HalluSEACurriculum, HalluSEASignal
from hallusea.converter import grit_gene_to_benchmark_item, grit_eval_result_to_verifier_input

__all__ = [
    "HalluSEACurriculum",
    "HalluSEASignal",
    "grit_gene_to_benchmark_item",
    "grit_eval_result_to_verifier_input",
]
