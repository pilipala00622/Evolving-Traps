# EvoHallu

一个面向知识库幻觉评测题生成的基础遗传算法框架。

这个版本只保留一条主线：

1. 从已有题目画像中抽取 `seed_questions.json`
2. 用四层基因组定义题目结构
3. 用基础 GA 做初始化、评估、交叉、变异和筛选
4. 通过真实 LLM API 或 mock 模式运行完整流程

## 设计目标

目标不是生成“更自然的 QA”，而是生成“更容易稳定诱发特定幻觉归因类型”的评测题。

四个基因维度必须和遗传算法步骤显式对应：

- `query`：问题结构
- `context`：上下文组织
- `trap`：幻觉陷阱
- `difficulty`：难度控制

在当前代码中：

- 交叉会按这四个维度逐层重组
- 变异级别会显式映射到这些维度
- 评估结果会回写到个体，供后续选择与分析使用

## 目录结构

```text
evo_hallucination/
├── config.py
├── llm.py
├── main.py
├── build_question_portraits.py
├── build_seed_prototypes.py
└── core/
    ├── gene.py
    ├── operators.py
    ├── fitness.py
    ├── llm_interface.py
    └── evolution.py
```

各文件职责如下：

- `config.py`：归因体系、任务类型、GA 超参数
- `llm.py`：统一真实模型调用接口
- `main.py`：基础 GA 主入口
- `build_question_portraits.py`：把多模型评测结果聚合成题目画像
- `build_seed_prototypes.py`：把题目画像压缩为可进化的种子配置
- `core/gene.py`：四层基因组和维度映射
- `core/operators.py`：选择、交叉、变异
- `core/fitness.py`：区分度 / 覆盖度 / 有效性评估
- `core/llm_interface.py`：实例化、作答、裁判评估、粗筛
- `core/evolution.py`：种群初始化、代际进化、最终筛选

## 运行方式

```bash
# Mock 模式
python3 main.py --mock

# 使用现成种子跑基础 GA
python3 main.py --seeds path/to/seed_questions.json

# 指定真实模型
python3 main.py \
  --seeds path/to/seed_questions.json \
  --generation-model gpt-5.1 \
  --judge-model gpt-5.1 \
  --eval-models gpt-5.2 deepseek-v3.2 ernie-5.0

# 额外覆盖部分 GA 参数
python3 main.py \
  --seeds path/to/seed_questions.json \
  --config path/to/ga_config.json
```

## 数据准备

```bash
python3 build_question_portraits.py --source-dir "第一阶段-14个模型-gpt51评测"
python3 build_seed_prototypes.py --portraits "第一阶段-14个模型-gpt51评测/question_portraits_detailed.jsonl"
```

## 注意

- `llm.py` 提供真实 API 调用能力，`core/llm_interface.py` 直接复用这套出口
- 当前仓库聚焦基础 GA 主线，不包含 agent 风格的额外探索/经验池机制
- 如果要做生产扩展，应优先保持四层基因和 GA 算子的对应关系不被打乱
