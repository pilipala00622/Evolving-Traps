# EvoHallu

一个面向知识库幻觉评测题生成的基础遗传算法框架。

这个版本只保留一条主线：

1. 从已有题目画像中抽取 `seed_questions.json`
2. 用四层基因组定义题目结构
3. 用基础 GA 做初始化、评估、交叉、变异和筛选
4. 通过真实 LLM API 或 mock 模式运行完整流程
5. 对最终候选题做自动验证和难度校准，形成可人工复核的 benchmark 候选集

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
- `core/benchmark_schema.py`：标准 benchmark item 数据结构
- `core/benchmark_validation.py`：候选题验证、人工复核建议、锚点模型校准

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

# 生成 benchmark 候选集（含自动验证与校准）
python3 main.py \
  --mock \
  --build-benchmark \
  --validation-repeats 3
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

## 逐步实现建议

当前版本已经补上了第一阶段底座：

1. `Individual` 仍是搜索空间里的候选题
2. `BenchmarkItem` 是准备入库的标准题结构
3. `BenchmarkValidator` 负责自动估计：
   - 目标错误触发率
   - 非目标错误泄漏率
   - 自然度/可作答率
   - 是否需要人工复核
4. `BenchmarkCalibrator` 负责用锚点模型给题目打经验难度分

### 哪些部分必须人工检查或标注

- 目标错误类型标注：
  自动统计只能给出建议，最终需要人工确认题目到底是不是在“主要诱导”目标错误，而不是混合错误。
- `real_time / out_of_date` 时间属性：
  需要人工确认知识截止时间、上下文时间戳、paired item 是否真的只改了时间条件。
- 参考答案：
  需要人工检查 `reference_answer` 是否完全由 `context` 支撑，避免把外部知识写进标答。
- 题目自然性：
  自动 naturalness 分数只能粗筛，正式 bench 仍建议人工抽检“像不像真实用户问题”。
- 分数校准阈值：
  `40-60` 的经验分段只是默认值，后续要根据你选定的 anchor models 和目标模型版本人工调整。

### 下一阶段最值得继续做的事

- 加 `paired benchmark builder`，专门生成 `real_time` 与 `out_of_date` 成对题
- 把 validator 的输出接到训练数据回流逻辑里
- 引入 item-level confusion matrix，显式区分目标错误与非目标错误

## 人工审核工具

仓库已经提供独立脚本 [review_benchmark_candidates.py](/Users/xyx/VscodeProjects/EvalBest/xyx_eval_diary/Eval_everything_in_my_era/hallucinate_eval_all/new_eval_for_IMA/attribution_agent/幻觉评测系统_v1.1/evo_hallucination/review_benchmark_candidates.py)：

```bash
python3 review_benchmark_candidates.py export \
  --input benchmark_candidates.json \
  --output benchmark_review_tasks.jsonl

python3 review_benchmark_candidates.py merge \
  --input benchmark_candidates.json \
  --reviews benchmark_review_tasks.reviewed.jsonl \
  --output benchmark_candidates.reviewed.json \
  --approved-output benchmark_release_candidates.json
```

详细审核规则、字段说明和“结果怎么喂回 bench / 训练 / 下一轮生成”见 [HUMAN_REVIEW_WORKFLOW.md](/Users/xyx/VscodeProjects/EvalBest/xyx_eval_diary/Eval_everything_in_my_era/hallucinate_eval_all/new_eval_for_IMA/attribution_agent/幻觉评测系统_v1.1/evo_hallucination/HUMAN_REVIEW_WORKFLOW.md)。
