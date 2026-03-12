"""
EvoHallu - 归因感知的遗传算法驱动幻觉评测题自动生产框架
配置文件：定义所有超参数、归因体系、领域列表等常量
"""

from dataclasses import dataclass, field
from typing import Dict, List

# ============================================================
# 归因体系定义 —— 这是整个框架的语义核心
# 每种归因类型对应一组可触发它的基因参数
# ============================================================

ATTRIBUTION_TYPES = {
    "错误匹配": {
        "id": "wrong_match",
        "description": "检索到了但配错了对象，将实体A的属性错误归给实体B",
        "trigger_genes": ["confusion_pairs", "semantic_similarity"],
        "severity": "high"
    },
    "引用错误": {
        "id": "wrong_citation",
        "description": "引用了文档中存在的章节，但章节内容不包含所声明的信息",
        "trigger_genes": ["doc_count", "distractor_ratio"],
        "severity": "high"
    },
    "限定错误": {
        "id": "scope_error",
        "description": "把某个限定范围内的属性泛化到更大范围",
        "trigger_genes": ["confusion_pairs", "hedging_level"],
        "severity": "high"
    },
    "缺证断言": {
        "id": "unsupported_claim",
        "description": "在缺乏充分证据的情况下输出超出可验证范围的断言",
        "trigger_genes": ["evidence_clarity", "info_gap"],
        "severity": "medium"
    },
    "确定性膨胀": {
        "id": "certainty_inflation",
        "description": "把推测性表达转为确定性断言",
        "trigger_genes": ["hedging_level", "evidence_clarity"],
        "severity": "medium"
    },
    "过度概括": {
        "id": "over_generalization",
        "description": "把个别案例推广为普遍规律",
        "trigger_genes": ["info_gap", "distractor_ratio"],
        "severity": "medium"
    },
    "引入新事实": {
        "id": "novel_fact",
        "description": "在生成任务中添加了原文未包含的信息",
        "trigger_genes": ["info_gap", "task_type"],
        "severity": "high"
    },
    "错误拼接": {
        "id": "wrong_merge",
        "description": "跨文档信息融合失败，将独立信息错误链接为因果关系",
        "trigger_genes": ["shared_entities", "cross_doc_overlap", "doc_count"],
        "severity": "high"
    },
}

# 任务类型 —— 对应报告中的"点线面体"认知递进
TASK_TYPES = {
    "信息定位": {
        "id": "info_locate",
        "level": 1,
        "description": "从给定文档中查找事实/字段/结论",
        "template": "{entity}的{attribute}是什么？"
    },
    "边界感知": {
        "id": "boundary",
        "level": 2,
        "description": "判断能否、是否、适用条件、范围限制",
        "template": "{entity}是否适用于{condition}的情况？"
    },
    "文档整合": {
        "id": "doc_integrate",
        "level": 3,
        "description": "跨段/跨文档信息整合、归纳、推断",
        "template": "结合相关文档，总结{topic}的{dimension}"
    },
    "生成控制": {
        "id": "generation",
        "level": 4,
        "description": "改写、润色、写作、提纲、文案生成",
        "template": "根据文档内容，以{format}撰写{content_type}"
    },
}

# 知识库领域
DOMAINS = [
    "经济金融", "健康医疗", "科技互联网", "教育考试", "法律政务",
    "传统行业", "工作职场", "自然科学", "娱乐休闲", "动物宠物",
    "农林牧渔", "出行交通", "文化历史", "家居生活", "其他"
]


# ============================================================
# 遗传算法超参数
# ============================================================

@dataclass
class GAConfig:
    """遗传算法核心配置"""

    # --- 种群参数 ---
    population_size: int = 200          # 种群大小（生产环境建议1500）
    elite_ratio: float = 0.3            # 精英保留比例
    max_generations: int = 20           # 最大进化代数
    convergence_patience: int = 5       # fitness连续N代不提升则停止

    # --- 适应度权重 ---
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        "discrimination": 0.5,          # 区分度权重
        "coverage": 0.3,                # 归因覆盖度权重
        "validity": 0.2,                # 有效性权重
    })
    redundancy_penalty: float = 0.15    # 重复度惩罚系数
    similarity_threshold: float = 0.85  # embedding去重阈值

    # --- 变异参数 ---
    mutation_rates: Dict[str, float] = field(default_factory=lambda: {
        "light": 0.60,                  # 轻度变异：query/trap/difficulty 微调
        "medium": 0.30,                 # 中度变异：context 重组 + trap 微调
        "heavy": 0.10,                  # 重度变异：query 或 context 结构切换
    })
    dynamic_mutation: bool = True       # 是否启用动态变异率
    mutation_boost_threshold: int = 3   # 连续N代不提升时提高变异率

    # --- 评估参数 ---
    eval_models: List[str] = field(default_factory=lambda: [
        "gpt-5.2",                      # 头部模型
        "deepseek-v3.2",                # 中部模型
        "ernie-5.0",                    # 尾部模型
    ])
    judge_model: str = "gpt-5.1"       # 裁判模型
    coarse_filter_model: str = "deepseek-v3.2"  # 粗筛只用1个模型
    validity_threshold: float = 3.0     # 自然度最低分（1-5）
    domain_quota: float = 0.15          # 单领域最大占比

    # --- 分阶段进化 ---
    phase1_generations: int = 10        # 前N代只优化区分度+有效性
    phase2_start_coverage: bool = True  # 第二阶段加入归因覆盖度

    # --- 目标 ---
    target_pool_size: int = 500         # 最终题库目标数量
