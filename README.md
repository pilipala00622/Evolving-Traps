# EvoHallu

这个仓库已经切换到新的唯一主线：

**围绕 3 类 hard hallucination，构建可控出题、人工审核和稳定诱发评测链路。**

当前只保留下面 3 类主目标：

1. `弱证据强结论`
2. `缺失关键信息下硬答`
3. `背景证据当直接证据`

这三类不是普通“难题”，而是更接近真正幻觉的高风险场景：
- 模型把弱证据说成强结论
- 文档缺关键信息，但模型仍然自信补全
- 文档有相关背景，但模型把背景材料误当成直接证据

---

## 新主线目标

目标不再是“尽可能把答案结构做成 pass/fail verifier”，而是：

- 生成真正像 hallucination trigger 的题
- 保证题面清楚，但证据边界危险
- 让人工审核重点判断：
  - 这题是否真在诱发幻觉
  - 幻觉方向是否单一
  - 模型正确行为是否应该是克制、保留或拒绝越界

---

## 新工作流

### Step 1：抽取纯知识库 context

从原始数据中提取第一轮 `user` 的纯文档内容：

```bash
python3 main.py extract-contexts
```

输出：
- `data/hard_hallucination/source_contexts.jsonl`

### Step 2：生成 hard hallucination 题卡

基于纯文档 context，用 `gpt-5.4` 生成三类 hard hallucination 题：

```bash
python3 main.py generate-cards
```

输出：
- `data/hard_hallucination/hard_hallucination_cards.jsonl`

### Step 3：构建 review 任务和标注页

```bash
python3 main.py build-review
```

输出：
- `data/hard_hallucination/review_tasks.jsonl`
- `data/hard_hallucination/review_annotation_studio.simple.html`

---

## 新数据结构

当前主线只关心下面几个字段：

- `query`
- `hard_hallucination_family`
- `target_failure_mode`
- `target_failure_subtype`
- `trigger_mechanism`
- `expected_safe_behavior`
- `why_hallucinatory`
- `evidence_source_hint`
- `judge_anchor`

不再把主线押在：
- `numeric_value / boolean_decision / entity_set / citation_set` 这些 carrier
- 以及旧版 baseline / challenge / verifier-heavy 产物

这些结构化 carrier 之后还能辅助分析，但不再作为主线定义。

---

## 3 类 hard hallucination 定义

详细定义见：
[hard_hallucination_标注协议_v1.md](/Users/xyx/VscodeProjects/EvalBest/xyx_eval_diary/Eval_everything_in_my_era/hallucinate_eval_all/new_eval_for_IMA/attribution_agent/幻觉评测系统_v1.1/evo_hallucination/reference/hard_hallucination_标注协议_v1.md)

简要理解：

- `弱证据强结论`
  文档里只有相关线索，没有足够直接支持，模型却下了确定结论。

- `缺失关键信息下硬答`
  文档缺关键字段，本该保留或说明“未提供”，模型却补出了不存在的信息。

- `背景证据当直接证据`
  文档里有背景或相关材料，但并不能直接支撑问题答案，模型却把它当成直接证据。

---

## 当前重要文件

### 方案

- [hard_hallucination_主线方案_v1.md](/Users/xyx/VscodeProjects/EvalBest/xyx_eval_diary/Eval_everything_in_my_era/hallucinate_eval_all/new_eval_for_IMA/attribution_agent/幻觉评测系统_v1.1/evo_hallucination/reference/hard_hallucination_主线方案_v1.md)
- [hard_hallucination_出题规范_v1.md](/Users/xyx/VscodeProjects/EvalBest/xyx_eval_diary/Eval_everything_in_my_era/hallucinate_eval_all/new_eval_for_IMA/attribution_agent/幻觉评测系统_v1.1/evo_hallucination/reference/hard_hallucination_出题规范_v1.md)
- [hard_hallucination_标注协议_v1.md](/Users/xyx/VscodeProjects/EvalBest/xyx_eval_diary/Eval_everything_in_my_era/hallucinate_eval_all/new_eval_for_IMA/attribution_agent/幻觉评测系统_v1.1/evo_hallucination/reference/hard_hallucination_标注协议_v1.md)

### 代码

- [main.py](/Users/xyx/VscodeProjects/EvalBest/xyx_eval_diary/Eval_everything_in_my_era/hallucinate_eval_all/new_eval_for_IMA/attribution_agent/幻觉评测系统_v1.1/evo_hallucination/main.py)
- [extract_request_contexts.py](/Users/xyx/VscodeProjects/EvalBest/xyx_eval_diary/Eval_everything_in_my_era/hallucinate_eval_all/new_eval_for_IMA/attribution_agent/幻觉评测系统_v1.1/evo_hallucination/extract_request_contexts.py)
- [generate_guided_queries.py](/Users/xyx/VscodeProjects/EvalBest/xyx_eval_diary/Eval_everything_in_my_era/hallucinate_eval_all/new_eval_for_IMA/attribution_agent/幻觉评测系统_v1.1/evo_hallucination/generate_guided_queries.py)
- [build_hard_hallucination_review_tasks.py](/Users/xyx/VscodeProjects/EvalBest/xyx_eval_diary/Eval_everything_in_my_era/hallucinate_eval_all/new_eval_for_IMA/attribution_agent/幻觉评测系统_v1.1/evo_hallucination/build_hard_hallucination_review_tasks.py)
- [generate_hard_hallucination_review_ui.py](/Users/xyx/VscodeProjects/EvalBest/xyx_eval_diary/Eval_everything_in_my_era/hallucinate_eval_all/new_eval_for_IMA/attribution_agent/幻觉评测系统_v1.1/evo_hallucination/generate_hard_hallucination_review_ui.py)
- [hard_hallucination_config.py](/Users/xyx/VscodeProjects/EvalBest/xyx_eval_diary/Eval_everything_in_my_era/hallucinate_eval_all/new_eval_for_IMA/attribution_agent/幻觉评测系统_v1.1/evo_hallucination/hard_hallucination_config.py)
- [core/hard_hallucination_schema.py](/Users/xyx/VscodeProjects/EvalBest/xyx_eval_diary/Eval_everything_in_my_era/hallucinate_eval_all/new_eval_for_IMA/attribution_agent/幻觉评测系统_v1.1/evo_hallucination/core/hard_hallucination_schema.py)

---

## 当前原则

- 不再继续沿旧 `baseline/challenge/verifier-heavy` 主线堆数据
- 先保证题真正像 hallucination trigger
- 再做人工审核
- 再做跨模型稳定诱发实验

如果一个问题只是抽取难、集合难、匹配难，但不体现“越界、自信补全、把背景当直接支持”，那它不属于当前主线。
