"""
LLM接口模块：将基因组"实例化"为实际的评测题目，并在需要时走真实 API。

核心功能：
1. 基因组 -> 实际的 query+context 文本（实例化）
2. 调用被测模型获取回答
3. 调用裁判模型进行归因评估
4. 评估题目自然度
"""

import json
import random
import re
from typing import Any, Dict, Optional, Tuple

from core.gene import Individual
from config import ATTRIBUTION_TYPES, TASK_TYPES
from llm import get_llm_instance


MODEL_NAME_ALIASES = {
    "ernie-5.0": "ernie-5.0-thinking",
    "qwen-3.5-plus": "qwen3.5-plus",
    "claude-sonnet-4-6": "claude-sonnet-4.6",
    "claude-opus-4-6": "claude-opus-4.6",
    "kimi-k2.5": "kimi-k2.5",
    "Kimi-K2.5": "kimi-k2.5",
}


class LLMInterface:
    """
    LLM调用的抽象层。
    
    生产环境中替换为实际的API调用（OpenAI / DeepSeek / 混元等）。
    当前实现为模拟器，用于验证整体流程的正确性。
    """

    def __init__(
        self,
        use_mock: bool = True,
        generation_model: str = "gpt-5.1",
        judge_model: str = "gpt-5.1",
        coarse_filter_model: str = "deepseek-v3.2",
    ):
        self.use_mock = use_mock
        self.generation_model = generation_model
        self.judge_model = judge_model
        self.coarse_filter_model = coarse_filter_model
        self._api_call_count = 0
        self._instantiation_cache: Dict[str, Dict[str, str]] = {}
        self._evaluation_cache: Dict[Tuple[str, str], Tuple[float, Dict[str, float]]] = {}
        self._validity_cache: Dict[str, Tuple[float, bool]] = {}

    @property
    def api_call_count(self) -> int:
        return self._api_call_count

    # ===========================================================
    # 1. 实例化：基因组 → 实际题目
    # ===========================================================

    def instantiate(self, individual: Individual, knowledge_base: Dict = None) -> Individual:
        """
        将基因组参数实例化为具体的query文本和context文本。
        
        这是遗传算法与LLM的核心接口：
        - 基因组定义了"什么样的题"（参数化描述）
        - LLM负责把参数化描述变成"具体的题"（自然语言）
        
        Prompt设计思路：
        把基因组的每个参数翻译为对LLM的约束条件，
        让LLM在约束范围内创造性地生成题目。
        """
        if self.use_mock:
            return self._mock_instantiate(individual)

        if individual.query_text and individual.context_text and individual.reference_answer:
            return individual

        cached = self._instantiation_cache.get(individual.id)
        if cached:
            individual.query_text = cached["query"]
            individual.context_text = cached["context"]
            individual.reference_answer = cached["reference_answer"]
            return individual

        prompt = self._build_instantiation_prompt(individual)
        response = self._call_json_model(
            self.generation_model,
            prompt,
            required_keys=("query", "context", "reference_answer"),
        )
        individual.query_text = str(response["query"]).strip()
        individual.context_text = str(response["context"]).strip()
        individual.reference_answer = str(response["reference_answer"]).strip()
        self._instantiation_cache[individual.id] = {
            "query": individual.query_text,
            "context": individual.context_text,
            "reference_answer": individual.reference_answer,
        }
        return individual

    def _build_instantiation_prompt(self, ind: Individual) -> str:
        """构建实例化prompt —— 将基因参数转为LLM的约束指令"""
        gene_vec = ind.get_gene_vector()
        target_attr = ind.trap_gene.target_attribution or "不限定"

        prompt = f"""你是一个幻觉评测题目设计专家。请根据以下参数化约束，生成一道知识库场景下的评测题目。

## 约束条件

### Query约束
- 任务类型: {gene_vec['task_type']}（{TASK_TYPES.get(gene_vec['task_type'], {}).get('description', '')}）
- 推理深度: 需要{gene_vec['complexity']}步推理
- 推理步骤扩展: {gene_vec['step_expansion']}步

### Context约束  
- 知识库领域: {gene_vec['domain']}
- 文档数量: {gene_vec['doc_count']}篇
- 文档间语义相似度: {gene_vec['semantic_similarity']:.1f}（0=完全不相关, 1=高度相似）
- 跨文档共享实体数: {gene_vec['shared_entities']}个
- 关键信息位置: {gene_vec['answer_position']}（head=开头, mid=中间, tail=末尾）
- 干扰段落占比: {gene_vec['distractor_ratio']:.0%}
- 总长度: 约{gene_vec['total_length']}字

### 陷阱设计约束（关键！）
- 混淆实体对数量: {gene_vec['confusion_pairs']}对（在文档中设置相似但不同的实体）
- 证据清晰度: {gene_vec['evidence_clarity']:.1f}（0=极模糊, 1=极清晰）
- 模糊词汇程度: {gene_vec['hedging_level']}级（0=无模糊词, 3=大量模糊词）
- 信息缺口程度: {gene_vec['info_gap']:.1f}（0=信息完整, 1=严重缺失）
- 跨文档重叠度: {gene_vec['cross_doc_overlap']:.1f}
- 目标归因类型: {target_attr}

## 输出格式（JSON）
```json
{{
    "query": "生成的用户问题",
    "context": "生成的知识库上下文（包含文档标题和内容）",
    "reference_answer": "基于context的标准答案",
    "trap_description": "陷阱设计说明（解释哪里容易出错）"
}}
```

要求：
1. query必须像真实用户会在知识库中提出的问题
2. context必须包含足够的信息来回答query（但可能包含干扰信息）
3. 陷阱设计必须符合约束参数，不要过于人工化
4. reference_answer必须严格基于context，不引入外部知识
5. 必须直接返回一个 JSON 对象，不要输出额外解释
"""
        return prompt

    def _mock_instantiate(self, ind: Individual) -> Individual:
        """模拟实例化（用于流程测试）"""
        gene = ind.get_gene_vector()
        ind.query_text = f"[{gene['domain']}][{gene['task_type']}] 模拟问题 (复杂度{gene['complexity']})"
        ind.context_text = f"[模拟上下文] 领域={gene['domain']}, 文档数={gene['doc_count']}, 长度≈{gene['total_length']}"
        ind.reference_answer = f"[模拟标准答案]"
        return ind

    # ===========================================================
    # 2. 被测模型评估：获取模型回答并检测幻觉
    # ===========================================================

    def evaluate_with_model(
        self,
        individual: Individual,
        model_name: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        用被测模型回答题目，然后用裁判模型检测幻觉。
        
        Returns:
            (幻觉率, {归因类型: 触发强度})
        """
        if self.use_mock:
            return self._mock_evaluate(individual, model_name)

        self.instantiate(individual)

        resolved_model = self._resolve_model_name(model_name)
        cache_key = (individual.id, resolved_model)
        if cache_key in self._evaluation_cache:
            return self._evaluation_cache[cache_key]

        answer_prompt = self._build_answer_prompt(individual)
        model_response = self._call_text_model(resolved_model, answer_prompt)

        judge_prompt = self._build_judge_prompt(individual, model_response, resolved_model)
        judge_result = self._call_json_model(
            self.judge_model,
            judge_prompt,
            required_keys=("hallucination_rate", "attributions"),
        )

        hallucination_rate = self._clamp_float(judge_result.get("hallucination_rate"), 0.0, 1.0)
        attributions = self._normalize_attributions(judge_result.get("attributions"))
        result = (hallucination_rate, attributions)
        self._evaluation_cache[cache_key] = result
        return result

    def _mock_evaluate(
        self,
        ind: Individual,
        model_name: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        模拟评估：基于基因参数生成合理的模拟结果。
        
        模拟逻辑：
        - 陷阱参数越强 → 幻觉率越高
        - 不同"模型"对不同陷阱的敏感度不同
        - 这确保了即使在模拟模式下，遗传算法也能正常进化
        """
        gene = ind.get_gene_vector()

        # 模型能力档位
        model_skill = {
            "gpt-5.2": 0.85,
            "deepseek-v3.2": 0.65,
            "ernie-5.0": 0.45,
        }.get(model_name, 0.6)

        # 基于陷阱参数计算基础幻觉率
        trap_difficulty = (
            gene["confusion_pairs"] * 0.08
            + (1 - gene["evidence_clarity"]) * 0.15
            + gene["hedging_level"] * 0.05
            + gene["info_gap"] * 0.12
            + gene["cross_doc_overlap"] * 0.10
            + gene["distractor_ratio"] * 0.05
        )

        # 模型能力调节
        hallucination_rate = trap_difficulty * (1.2 - model_skill)
        hallucination_rate = max(0, min(1, hallucination_rate + random.gauss(0, 0.05)))

        # 模拟归因分布（基于哪些陷阱参数最强）
        attributions = {}
        if gene["confusion_pairs"] >= 2:
            attributions["错误匹配"] = gene["confusion_pairs"] / 5 * random.uniform(0.6, 1.0)
            attributions["限定错误"] = gene["confusion_pairs"] / 5 * random.uniform(0.3, 0.7)
        if gene["evidence_clarity"] < 0.5:
            attributions["缺证断言"] = (1 - gene["evidence_clarity"]) * random.uniform(0.5, 1.0)
        if gene["hedging_level"] >= 2:
            attributions["确定性膨胀"] = gene["hedging_level"] / 3 * random.uniform(0.5, 0.9)
        if gene["info_gap"] > 0.4:
            attributions["引入新事实"] = gene["info_gap"] * random.uniform(0.4, 0.9)
        if gene["cross_doc_overlap"] > 0.5 and gene["shared_entities"] >= 2:
            attributions["错误拼接"] = gene["cross_doc_overlap"] * random.uniform(0.5, 1.0)

        # 确保至少有一个归因类型
        if not attributions:
            attributions["缺证断言"] = random.uniform(0.1, 0.3)

        return hallucination_rate, attributions

    # ===========================================================
    # 3. 有效性评估：自然度 + 可作答性
    # ===========================================================

    def evaluate_validity(self, individual: Individual) -> Tuple[float, bool]:
        """
        评估题目的自然度和可作答性。
        
        Returns:
            (自然度评分1-5, 是否可作答)
        """
        if self.use_mock:
            return self._mock_validity(individual)

        self.instantiate(individual)

        if individual.id in self._validity_cache:
            return self._validity_cache[individual.id]

        prompt = self._build_validity_prompt(individual)
        response = self._call_json_model(
            self.judge_model,
            prompt,
            required_keys=("naturalness_score", "is_answerable"),
        )

        naturalness = self._clamp_float(response.get("naturalness_score"), 1.0, 5.0, default=3.0)
        is_answerable = self._coerce_bool(response.get("is_answerable"))
        result = (naturalness, is_answerable)
        self._validity_cache[individual.id] = result
        return result

    def _mock_validity(self, ind: Individual) -> Tuple[float, bool]:
        """模拟有效性评估"""
        gene = ind.get_gene_vector()

        # 简单启发式：过于极端的参数组合会降低自然度
        naturalness = 4.0
        if gene["confusion_pairs"] > 4:
            naturalness -= 0.5
        if gene["info_gap"] > 0.8:
            naturalness -= 0.5
        if gene["distractor_ratio"] > 0.7:
            naturalness -= 0.3
        naturalness += random.gauss(0, 0.2)
        naturalness = max(1.0, min(5.0, naturalness))

        is_answerable = gene["info_gap"] < 0.9

        return naturalness, is_answerable

    # ===========================================================
    # 4. 粗筛评估（降低成本）
    # ===========================================================

    def coarse_evaluate(self, individual: Individual) -> bool:
        """
        粗筛：只用1个模型快速判断是否值得精评。
        
        通过条件：
        1. 有效性 ≥ 阈值
        2. 至少触发1个幻觉句
        
        不通过的个体直接淘汰，节省后续3-4个模型的API调用。
        """
        naturalness, is_answerable = self.evaluate_validity(individual)

        if naturalness < 3.0 or not is_answerable:
            return False

        # 用粗筛模型快速检查是否能触发幻觉
        halluc_rate, _ = self.evaluate_with_model(
            individual, self.coarse_filter_model
        )

        # 幻觉率>0就通过粗筛（说明这道题至少有检测价值）
        return halluc_rate > 0.05

    # ===========================================================
    # 真实 API 调用辅助方法
    # ===========================================================

    def _resolve_model_name(self, model_name: str) -> str:
        return MODEL_NAME_ALIASES.get(model_name, model_name)

    def _call_text_model(self, model_name: str, prompt: str) -> str:
        resolved_name = self._resolve_model_name(model_name)
        llm = get_llm_instance(resolved_name)
        response = llm.get_model_answer(prompt, history=[])
        self._api_call_count += 1
        return response

    def _call_json_model(
        self,
        model_name: str,
        prompt: str,
        required_keys: Tuple[str, ...] = (),
    ) -> Dict[str, Any]:
        raw_response = self._call_text_model(model_name, prompt)
        parsed = self._extract_json_object(raw_response)
        if not isinstance(parsed, dict):
            raise ValueError(f"{model_name} 未返回可解析的 JSON 对象: {raw_response[:300]}")
        for key in required_keys:
            if key not in parsed:
                raise ValueError(f"{model_name} 返回缺少字段 {key}: {parsed}")
        return parsed

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return None

        direct = self._try_json_loads(text)
        if isinstance(direct, dict):
            return direct

        fenced_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
        for block in fenced_blocks:
            parsed = self._try_json_loads(block)
            if isinstance(parsed, dict):
                return parsed

        candidate = self._extract_balanced_json(text)
        if candidate:
            parsed = self._try_json_loads(candidate)
            if isinstance(parsed, dict):
                return parsed

        return None

    def _extract_balanced_json(self, text: str) -> Optional[str]:
        start = text.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(text)):
                ch = text[idx]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:idx + 1]
            start = text.find("{", start + 1)
        return None

    def _try_json_loads(self, text: str) -> Optional[Any]:
        try:
            return json.loads(text)
        except Exception:
            return None

    def _normalize_attributions(self, value: Any) -> Dict[str, float]:
        normalized: Dict[str, float] = {}
        if isinstance(value, dict):
            for attr in ATTRIBUTION_TYPES:
                if attr in value:
                    normalized[attr] = self._clamp_float(value[attr], 0.0, 1.0)
        return {k: v for k, v in normalized.items() if v > 0}

    def _clamp_float(
        self,
        value: Any,
        min_value: float,
        max_value: float,
        default: float = 0.0,
    ) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = default
        return max(min_value, min(max_value, numeric))

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            return normalized in {"true", "1", "yes", "y", "是", "可作答", "可以"}
        return bool(value)

    def _build_answer_prompt(self, ind: Individual) -> str:
        return f"""你是知识库问答模型。请仅根据给定 context 回答问题，不要使用外部知识。

如果 context 证据不足，请明确说明“根据给定材料无法确定”，不要补充新事实。

## Query
{ind.query_text}

## Context
{ind.context_text}

请直接给出最终回答。"""

    def _build_judge_prompt(self, ind: Individual, model_response: str, model_name: str) -> str:
        attr_lines = "\n".join(
            f"- {name}: {meta['description']}" for name, meta in ATTRIBUTION_TYPES.items()
        )
        return f"""你是幻觉评测裁判。请判断给定模型回答中，有多少内容超出了 context 和 reference_answer 可支持的范围，并给出归因强度。

请特别依据以下归因定义判断：
{attr_lines}

输出一个 JSON 对象，不要输出额外说明：
{{
  "hallucination_rate": 0.0,
  "attributions": {{
    "错误匹配": 0.0,
    "引用错误": 0.0,
    "限定错误": 0.0,
    "缺证断言": 0.0,
    "确定性膨胀": 0.0,
    "过度概括": 0.0,
    "引入新事实": 0.0,
    "错误拼接": 0.0
  }},
  "notes": "一句话说明主要问题"
}}

要求：
1. `hallucination_rate` 取值 0 到 1，表示回答中存在幻觉或不可证实内容的比例。
2. `attributions` 里每个值都在 0 到 1 之间；未触发填 0。
3. 只有当 context 或 reference_answer 能明确支持时，才算非幻觉。

## 被测模型
{model_name}

## Query
{ind.query_text}

## Context
{ind.context_text}

## Reference Answer
{ind.reference_answer}

## Model Response
{model_response}
"""

    def _build_validity_prompt(self, ind: Individual) -> str:
        return f"""你是评测题质量审核员。请判断下面这道题是否自然、是否能仅基于给定 context 回答。

输出一个 JSON 对象，不要输出额外说明：
{{
  "naturalness_score": 4.0,
  "is_answerable": true,
  "notes": "一句话说明"
}}

要求：
1. `naturalness_score` 范围是 1 到 5。
2. `is_answerable` 只表示“给定 context 是否足以回答 query”，不要考虑外部知识。
3. 若 context 信息不足、题面不自然或明显人工拼接，请如实降低分数。

## Query
{ind.query_text}

## Context
{ind.context_text}

## Reference Answer
{ind.reference_answer}
"""
