# EvoHallu / GRIT

面向企业噪声 RAG 场景的结构化幻觉诱发与基因进化框架。

这个项目不把 hallucination 看成随机噪声，而是把它建模为可以被发现、诱发、归档和持续扩展的**失败机制**。当前主线由三层组成：

- `HalluBench-lite`：基于真实企业知识库上下文和人工种子卡片构建的 benchmark 基底
- `GRIT`：`Genetic search for Reliable Induction of Targeted hallucinations`，档案驱动的基因进化搜索框架
- `HalluSEA`：将稳定失败模式转化为未来可验证奖励训练信号的框架接口

## 核心问题

企业知识库问答系统中的 RAG 检索经常并不“干净”：

- 检索结果可能只提供弱相关线索，不能直接支持强结论
- 关键字段可能缺失，但模型仍被精确回答压力推着往前答
- 上下文可能只是背景材料，却被模型误当成直接证据

因此本项目的目标不是生成“难题”，而是生成**证据边界危险、容易诱发越界回答的高风险题**。

## 双轴分类体系

每条样本沿两个独立维度定义。

### 1. 机制轴

| 机制 | 描述 | 正确行为 |
|------|------|---------|
| `弱证据强结论` | 文档只有相关线索，不足以直接支持结论，模型却给出确定性判断 | 保留判断，明确证据不足 |
| `缺失关键信息下硬答` | 文档缺关键字段，模型编造数值、实体或引文 | 明确说明文档未提供，拒绝补全 |
| `背景证据当直接证据` | 文档提供背景材料，但不足以直接作答，模型误当直接支撑 | 区分背景与直接证据，不越界推断 |

### 2. 载体轴

| 载体 | 含义 |
|------|------|
| `numeric` | 数值型答案 |
| `boolean` | 是/否或判断型答案 |
| `entity_set` | 实体集合型答案 |
| `citation_set` | 引文集合型答案 |

这种分离很关键：同一失败机制可以落在不同答案载体上，同一种载体也可能承载不同失败机制。

## 自动评测标签

每条模型输出被归入四个互斥标签之一：

- `correct`
- `target_error`
- `non_target_error`
- `unparseable`

目标不是泛泛地“让模型答错”，而是**尽可能高概率地诱发目标错误，同时压低非目标错误泄漏**。

## 三层方法结构

### HalluBench-lite：人工种子层

从真实企业知识库上下文出发，由人工标注者编写少量高价值种子卡片。每张卡片描述：

- 源上下文和支撑边界
- 题目中的冲突点或证据缺口
- 正确的保守行为
- 目标错误类型
- 答案载体类型
- 为什么这类样本可能稳定诱发目标机制

这一步不追求大规模，而是追求**小而准的失败机制原型**。

### GRIT：基因归纳与档案驱动进化

系统不会直接从种子扩题，而是先让 LLM 把种子抽象成可复用的“基因”。每条基因是一个关于“如何设计这类幻觉触发场景”的语义规格，而不是具体题目。

典型基因字段包括：

| 字段 | 含义 |
|------|------|
| `task_frame` | 企业任务类型 |
| `failure_mechanism` | 目标失败机制 |
| `trigger_form` | 缺口或歧义在题面中的呈现方式 |
| `support_gap_type` | 缺失的关键信息类型 |
| `target_error_type` | 目标错误标签 |
| `answer_carrier` | 答案载体类型 |
| `abstention_expected` | 正确行为是否应为拒答/克制 |
| `difficulty_knobs` | 可调难度参数 |
| `mutation_axes` | 可以沿哪些方向变异 |

有了基因之后，搜索围绕“机制”而不是“措辞”展开，允许系统积累跨轮次可复用的失败谱系档案。

### HalluSEA：训练接口层

当前仓库中的 `HalluSEA` 仍是框架与先导协议，不是已经完成的大规模训练系统。它的作用是把稳定、高纯度、可类型化的失败模式转成后续训练可使用的奖励接口和课程单元。

## 评测指标

项目当前使用的主指标来自论文中的定义：

```text
F(g) = 0.45 * TEHR + 0.30 * SIS@2/3 + 0.25 * Purity - triviality_penalty
```

| 指标 | 含义 |
|------|------|
| `TEHR` | `Target Error Hit Rate`，目标错误命中率 |
| `SIS@2/3` | `Stable Induction Score`，至少在 3 个评测模型中的 2 个上触发目标错误的稳定诱发比例 |
| `Purity` | `target_error / (target_error + non_target_error)` |
| `Judgeable` | 非 `unparseable` 的可判定比例 |
| `triviality_penalty` | 防止候选题退化为对源题的简单改写 |

高适应度基因意味着：

- 能稳定命中目标错误机制
- 在多模型面板上具有跨模型一致性
- 诱发结果干净，不大量泄漏到非目标错误

## 进化流程

整体流程如下：

```text
人工种子
  -> 基因归纳
  -> 基因扩展为候选题
  -> 固定多模型面板评测
  -> 计算 TEHR / SIS / Purity
  -> 精英基因入档案
  -> 低分基因参考精英基因进行语义反思变异
  -> 生成 child gene
  -> 继续扩题 / 评测 / 归档
```

这里的“变异”不是随机扰动数值，而是**LLM 语义变异**：

- 让模型分析低分基因为什么失败
- 参考高分精英基因的结构特征
- 沿 `mutation_axes` 生成更强的 `child gene`

当前支持的变异 profile：

- `general`：通用进化，提升 `TEHR / SIS / Purity`
- `numeric_fabrication`：数值型专项 profile，固定 `answer_carrier=numeric`，强制制造关键口径缺失与精确数值压力

## 当前实验设定与结果

根据 `main_CN.tex`，当前论文版本的核心结果如下：

- 从 `9` 张人工种子卡片出发，覆盖 `3` 类机制和 `boolean / numeric / entity_set` 三类载体
- 共运行 `6` 轮进化
- 最终形成 `21` 个有效归档基因
- 固定评测面板为 `Qwen3.5-Plus`、`HunYuan-2.0-Thinking`、`Gemini-3.1-Pro`

关键结果：

- 第 3 轮候选题达到 `SIS@2/3 = 0.875`
- 主稳定诱发切片 `8` 条，达到 `SIS@2/3 = 1.0`，`Purity = 1.0`
- 诊断切片 `8` 条，揭示明显模型差异，Qwen 更保守
- 数值型专项 `numeric_fabrication` profile 将数值无中生有分支提升到 `SIS@2/3 = 0.8333`

这些结果支持一个核心结论：**企业噪声 RAG 中的幻觉是结构化的、可诱发的，并且在一定程度上具有跨模型稳定性。**

## 仓库工作流

下面是和当前仓库脚本对应的主线流程。

### Step 0：准备 source contexts

```bash
python3 main.py extract-contexts
# 输出: data/hard_hallucination/source_contexts.jsonl
```

### Step 1：生成初始 hard hallucination 题卡

```bash
python3 main.py generate-cards
# 输出: data/hard_hallucination/hard_hallucination_cards.jsonl
```

### Step 2：人工审核 seed

```bash
python3 main.py build-review
# 输出: data/hard_hallucination/review/review_tasks.jsonl
# 输出: data/hard_hallucination/review/review_annotation_studio.simple.html
```

通过审核的 seed 才进入后续基因抽取与进化。

### Step 3：抽取语义基因

```bash
python3 extract_seed_genes.py \
  --input data/hard_hallucination/reviewed_seeds.jsonl \
  --output data/genes/seed_genes.jsonl
```

### Step 4：展开为候选题

```bash
python3 expand_genes_to_candidates.py \
  --genes data/genes/seed_genes.jsonl \
  --output data/candidates/
```

也可以直接结合真实知识库 context 做诱发：

```bash
python3 induce_from_source_contexts.py \
  --contexts data/hard_hallucination/source_contexts.jsonl \
  --genes data/genes/seed_genes.jsonl \
  --output data/candidates/induction_results.jsonl
```

### Step 5：多模型评测

```bash
python3 evaluate_hard_hallucination_candidates.py \
  --candidates data/candidates/ \
  --output data/eval_results/
```

### Step 6：基因进化

```bash
python3 run_gene_evolution.py \
  --genes data/genes/seed_genes.jsonl \
  --candidates data/candidates/ \
  --eval-results data/eval_results/ \
  --output data/genes/gen2_mutated.jsonl \
  --generation 1
```

重复 Step 4 到 Step 6，直到基因适应度和 archive 质量趋于稳定。

### Step 7：构建 benchmark 切片与发布包

```bash
python3 normalize_gene_bank.py
python3 merge_gene_archive.py
python3 aggregate_successful_inductions.py
python3 build_benchmark_slices.py
python3 package_benchmark_release.py
```

## 数据与结构化字段

当前主线最关心的字段包括：

- `query`
- `failure_mechanism`
- `target_error_type`
- `answer_carrier`
- `expected_safe_behavior`
- `support_gap_type`
- `trigger_form`
- `why_hallucinatory`
- `judge_anchor`

附录中定义的关键 schema 还包括：

- 种子字段：`seed_id`、`trace_id`、`pattern_name`、`mechanism`、`source_query`、`conflict_point`、`correct_behavior`、`target_error_type`、`answer_carrier`、`why_likely_to_fail`
- 基因字段：`gene_id`、`seed_id`、`trace_id`、`task_frame`、`failure_mechanism`、`trigger_form`、`support_gap_type`、`target_error_type`、`answer_carrier`、`abstention_expected`、`difficulty_knobs`、`verifier_shape`、`mutation_axes`、`generation`、`parent_gene_ids`、`fitness_history`

## 哪些部分必须人工检查

自动化评测能显著提速，但以下环节仍建议人工确认：

- `target_error` 是否真的命中了目标机制，而不是混合错误
- `reference_answer` 是否完全由给定 `context` 支撑
- 题目是否自然、像真实企业用户问题
- 时间敏感样本是否存在知识截止时间错配
- benchmark 切片是否保留了预期的机制和载体分布

## 局限与下一步

当前版本仍是先导验证阶段：

- `citation_set` 载体已进入分类体系，但还没有像其他载体那样成熟
- 三模型面板只能说明在固定面板内的稳定性，不能代表所有模型
- `HalluSEA` 目前是训练接口与课程框架，不是完整验证过的强化学习系统

下一步主要有两个方向：

- 扩展 `HalluBench-lite`，覆盖更丰富的领域和载体类型
- 将 `HalluSEA` 真正实例化为以 GRIT 档案为课程单元的可验证奖励训练系统

## 原则

- 先保证题目是真正的 hallucination trigger，再追求规模化扩张
- 不把“抽取难”“匹配难”误当成“幻觉诱发”
- 基因进化的目标不是题目数量，而是得到可复用、高纯度、跨模型稳定的失败机制模板
