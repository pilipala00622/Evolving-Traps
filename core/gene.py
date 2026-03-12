"""
基因编码模块：定义评测题的四层基因组，以及 GA 与基因维度之间的显式映射。

四个基因维度：
1. query      - 问题结构
2. context    - 知识库上下文组织方式
3. trap       - 幻觉诱发陷阱
4. difficulty - 难度控制
"""

import copy
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from config import DOMAINS, TASK_TYPES


GENE_LAYERS: Tuple[str, ...] = ("query", "context", "trap", "difficulty")
GENE_ATTR_MAP: Dict[str, str] = {
    "query": "query_gene",
    "context": "context_gene",
    "trap": "trap_gene",
    "difficulty": "difficulty_gene",
}

# 让 GA 的变异级别与基因维度显式对应，而不是散落在多个文件里。
MUTATION_LEVEL_TO_LAYERS: Dict[str, Tuple[str, ...]] = {
    "light": ("query", "trap", "difficulty"),
    "medium": ("context", "trap"),
    "heavy": ("query", "context"),
}


@dataclass
class QueryGene:
    """G1: Query模板基因 —— 定义问题的结构和意图"""

    task_type: str              # 任务类型: 信息定位/边界感知/文档整合/生成控制
    entity_slot: str = ""       # 实体槽位（进化时可替换）
    attribute_slot: str = ""    # 属性槽位
    condition_slot: str = ""    # 条件槽位（边界感知用）
    complexity: int = 1         # 推理深度: 1=单跳, 2=双跳, 3=多跳

    def mutate_light(self):
        """轻度变异：只替换 query 内部槽位，保持任务结构不变"""
        self.entity_slot = f"entity_{random.randint(1, 1000)}"
        self.attribute_slot = f"attr_{random.randint(1, 100)}"
        if self.task_type == "边界感知":
            self.condition_slot = f"condition_{random.randint(1, 200)}"

    def mutate_heavy(self):
        """重度变异：切换 query 的任务类型"""
        other_types = [t for t in TASK_TYPES if t != self.task_type]
        self.task_type = random.choice(other_types)
        self.complexity = random.randint(1, 3)


@dataclass
class ContextGene:
    """G2: Context组装基因 —— 定义文档选择与拼接策略"""

    domain: str                         # 知识库领域
    doc_count: int = 3                  # 文档数量 (1~15)
    semantic_similarity: float = 0.5    # 文档间语义相似度 (0~1)
    shared_entities: int = 2            # 跨文档共享实体数 (0~10)
    answer_position: str = "mid"        # 关键信息位置: head/mid/tail
    distractor_ratio: float = 0.3       # 干扰段落占比 (0~0.8)
    total_length: int = 30000           # 上下文总字符数 (3k~128k)
    doc_ids: List[str] = field(default_factory=list)  # 具体文档ID

    def mutate_medium(self):
        """中度变异：调整 context 维度的组装策略"""
        mutations = [
            lambda: setattr(self, 'doc_count', max(1, min(15, self.doc_count + random.choice([-2, -1, 1, 2])))),
            lambda: setattr(self, 'semantic_similarity', max(0, min(1, self.semantic_similarity + random.uniform(-0.2, 0.2)))),
            lambda: setattr(self, 'shared_entities', max(0, min(10, self.shared_entities + random.choice([-1, 1])))),
            lambda: setattr(self, 'answer_position', random.choice(["head", "mid", "tail"])),
            lambda: setattr(self, 'distractor_ratio', max(0, min(0.8, self.distractor_ratio + random.uniform(-0.15, 0.15)))),
            lambda: setattr(self, 'total_length', random.choice([5000, 15000, 30000, 50000, 80000, 120000])),
        ]
        for m in random.sample(mutations, k=random.randint(2, 3)):
            m()

    def mutate_heavy(self):
        """重度变异：切换 context 所处领域"""
        other_domains = [d for d in DOMAINS if d != self.domain]
        self.domain = random.choice(other_domains)


@dataclass
class TrapGene:
    """G3: 陷阱配置基因 —— 定义幻觉触发条件
    
    每个参数都直接关联归因体系中的某种幻觉类型。
    这是EvoLLMs论文中完全缺失的部分：归因感知的基因设计。
    """

    # 触发"错误匹配"和"限定错误"
    confusion_pairs: int = 2            # 混淆实体对数量 (0~5)

    # 触发"缺证断言"和"确定性膨胀"
    evidence_clarity: float = 0.7       # 证据清晰度 (0=极模糊, 1=极清晰)
    hedging_level: int = 1              # 文档模糊词汇程度 (0=无, 1=轻, 2=中, 3=重)

    # 触发"引入新事实"
    info_gap: float = 0.3               # 信息缺口程度 (0=完整, 1=严重缺失)

    # 触发"错误拼接"
    cross_doc_overlap: float = 0.4      # 跨文档信息重叠度 (0~1)

    # 目标归因类型（进化时可定向优化）
    target_attribution: str = ""        # 留空=不限定，否则指定归因类型

    def mutate_light(self):
        """轻度变异：微调 trap 维度参数"""
        self.confusion_pairs = max(0, min(5, self.confusion_pairs + random.choice([-1, 0, 1])))
        self.evidence_clarity = max(0, min(1, self.evidence_clarity + random.uniform(-0.1, 0.1)))
        self.hedging_level = max(0, min(3, self.hedging_level + random.choice([-1, 0, 1])))
        self.info_gap = max(0, min(1, self.info_gap + random.uniform(-0.1, 0.1)))
        self.cross_doc_overlap = max(0, min(1, self.cross_doc_overlap + random.uniform(-0.1, 0.1)))


@dataclass
class DifficultyGene:
    """G4: 难度参数基因"""

    target_difficulty: float = 0.5      # 目标难度 (0~1, 对应IRT的theta)
    step_expansion: int = 1             # 推理步数要求 (1~5)

    def mutate_light(self):
        """轻度变异：微调 difficulty 维度参数"""
        self.target_difficulty = max(0, min(1, self.target_difficulty + random.uniform(-0.1, 0.1)))
        self.step_expansion = max(1, min(5, self.step_expansion + random.choice([-1, 0, 1])))


# ============================================================
# 完整个体：一道评测题的全部基因组
# ============================================================

@dataclass
class Individual:
    """一道评测题 = 一个遗传个体"""

    # 四层基因组
    query_gene: QueryGene
    context_gene: ContextGene
    trap_gene: TrapGene
    difficulty_gene: DifficultyGene

    # 实例化后的实际内容（由LLM根据基因组生成）
    query_text: str = ""
    context_text: str = ""
    reference_answer: str = ""

    # 适应度评估结果
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    total_fitness: float = 0.0

    # 各模型在此题上的幻觉归因分布
    model_attributions: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # 元信息
    generation: int = 0                 # 诞生于第几代
    parent_ids: List[str] = field(default_factory=list)
    _id: str = ""

    def __post_init__(self):
        if not self._id:
            self._id = self._generate_id()

    def _generate_id(self) -> str:
        """基于基因组内容生成唯一ID"""
        content = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()[:12]

    @property
    def id(self) -> str:
        return self._id

    def gene_layers(self) -> Dict[str, object]:
        """返回按维度命名的基因层，用于让 GA 操作与基因维度一一对应。"""
        return {
            layer: getattr(self, attr_name)
            for layer, attr_name in GENE_ATTR_MAP.items()
        }

    def get_gene_vector(self) -> Dict:
        """将基因组展平为特征向量（用于相似度计算和分析）"""
        return {
            "task_type": self.query_gene.task_type,
            "entity_slot": self.query_gene.entity_slot,
            "attribute_slot": self.query_gene.attribute_slot,
            "condition_slot": self.query_gene.condition_slot,
            "complexity": self.query_gene.complexity,
            "domain": self.context_gene.domain,
            "doc_count": self.context_gene.doc_count,
            "semantic_similarity": self.context_gene.semantic_similarity,
            "shared_entities": self.context_gene.shared_entities,
            "answer_position": self.context_gene.answer_position,
            "distractor_ratio": self.context_gene.distractor_ratio,
            "total_length": self.context_gene.total_length,
            "confusion_pairs": self.trap_gene.confusion_pairs,
            "evidence_clarity": self.trap_gene.evidence_clarity,
            "hedging_level": self.trap_gene.hedging_level,
            "info_gap": self.trap_gene.info_gap,
            "cross_doc_overlap": self.trap_gene.cross_doc_overlap,
            "target_attribution": self.trap_gene.target_attribution,
            "target_difficulty": self.difficulty_gene.target_difficulty,
            "step_expansion": self.difficulty_gene.step_expansion,
        }

    def reset_runtime_state(
        self,
        generation: Optional[int] = None,
        parent_ids: Optional[List[str]] = None,
    ) -> "Individual":
        """重置实例化内容和评估结果，供新个体/子代复用。"""
        self.query_text = ""
        self.context_text = ""
        self.reference_answer = ""
        self.fitness_scores = {}
        self.total_fitness = 0.0
        self.model_attributions = {}
        if generation is not None:
            self.generation = generation
        if parent_ids is not None:
            self.parent_ids = parent_ids
        self._id = self._generate_id()
        return self

    def clone(self) -> "Individual":
        return copy.deepcopy(self)

    def dominant_attribution_type(self) -> Optional[str]:
        """返回此题最常触发的归因类型"""
        if not self.model_attributions:
            return None
        # 聚合所有模型的归因分布
        agg = {}
        for model_attrs in self.model_attributions.values():
            for attr_type, score in model_attrs.items():
                agg[attr_type] = agg.get(attr_type, 0) + score
        if not agg:
            return None
        return max(agg, key=agg.get)


# ============================================================
# 个体工厂：创建随机个体 / 从种子题创建变体
# ============================================================

class IndividualFactory:
    """个体工厂：负责创建初始种群和从种子题生成变体"""

    @staticmethod
    def create_random(domain: str = None, task_type: str = None) -> Individual:
        """创建一个完全随机的个体"""
        domain = domain or random.choice(DOMAINS)
        task_type = task_type or random.choice(list(TASK_TYPES.keys()))

        return Individual(
            query_gene=QueryGene(
                task_type=task_type,
                complexity=random.randint(1, 3),
            ),
            context_gene=ContextGene(
                domain=domain,
                doc_count=random.randint(1, 10),
                semantic_similarity=random.uniform(0.2, 0.9),
                shared_entities=random.randint(0, 5),
                answer_position=random.choice(["head", "mid", "tail"]),
                distractor_ratio=random.uniform(0.1, 0.6),
                total_length=random.choice([5000, 15000, 30000, 50000, 80000, 120000]),
            ),
            trap_gene=TrapGene(
                confusion_pairs=random.randint(0, 4),
                evidence_clarity=random.uniform(0.3, 1.0),
                hedging_level=random.randint(0, 3),
                info_gap=random.uniform(0.0, 0.6),
                cross_doc_overlap=random.uniform(0.0, 0.7),
            ),
            difficulty_gene=DifficultyGene(
                target_difficulty=random.uniform(0.2, 0.8),
                step_expansion=random.randint(1, 3),
            ),
        )

    @staticmethod
    def create_from_seed(seed_config: Dict) -> Individual:
        """从现有150题的配置创建个体（种子 → 初始种群）"""
        return Individual(
            query_gene=QueryGene(**seed_config.get("query", {})),
            context_gene=ContextGene(**seed_config.get("context", {})),
            trap_gene=TrapGene(**seed_config.get("trap", {})),
            difficulty_gene=DifficultyGene(**seed_config.get("difficulty", {})),
        )

    @staticmethod
    def create_variant(parent: Individual, mutation_type: str = "light") -> Individual:
        """从父代创建一个变体（用于初始种群扩充）"""
        child = copy.deepcopy(parent)
        child.reset_runtime_state(parent_ids=[parent.id])
        apply_mutation_level(child, mutation_type)
        child._id = child._generate_id()
        return child


def iter_gene_layers(individual: Individual) -> Iterable[Tuple[str, object]]:
    """按固定顺序遍历个体的四个基因维度。"""
    for layer in GENE_LAYERS:
        yield layer, getattr(individual, GENE_ATTR_MAP[layer])


def build_individual_from_layers(
    layers: Dict[str, object],
    generation: int = 0,
    parent_ids: Optional[List[str]] = None,
) -> Individual:
    """根据四个基因维度重建个体，避免交叉逻辑里硬编码位置索引。"""
    return Individual(
        query_gene=copy.deepcopy(layers["query"]),
        context_gene=copy.deepcopy(layers["context"]),
        trap_gene=copy.deepcopy(layers["trap"]),
        difficulty_gene=copy.deepcopy(layers["difficulty"]),
        generation=generation,
        parent_ids=parent_ids or [],
    )


def apply_mutation_level(individual: Individual, mutation_level: str) -> None:
    """
    将 GA 里的变异级别显式映射到基因维度。

    - light: 细调 query/trap/difficulty
    - medium: 调整 context，并同步微调 trap
    - heavy: 重构 query 或 context 结构
    """
    if mutation_level == "light":
        individual.query_gene.mutate_light()
        individual.trap_gene.mutate_light()
        individual.difficulty_gene.mutate_light()
        return

    if mutation_level == "medium":
        individual.context_gene.mutate_medium()
        individual.trap_gene.mutate_light()
        return

    if mutation_level == "heavy":
        if random.random() < 0.5:
            individual.query_gene.mutate_heavy()
        else:
            individual.context_gene.mutate_heavy()
        return

    raise ValueError(f"未知的 mutation_level: {mutation_level}")
