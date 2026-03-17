# Query / Response / Sentence 分层 Schema 草案

## 1. 为什么要改

旧口径默认把 `query` 直接贴成某个“归因标签”。

这会有一个根本问题：

- 真实归因通常发生在 `response` 中的具体句子
- 而不是发生在 `query` 本身

所以新的口径是：

- `query-level` 标“诱发目标 / intended failure mode”
- `response-level` 标“这次回答整体主要出了什么问题”
- `sentence-level` 标“真实发生的归因”

---

## 2. 新的三层对象

### QueryItem

作用：

- 存储题面和上下文
- 表示“这道题想诱发什么错误”
- 不再假设 query 自己就有真实归因标签

核心字段：

- `query_id`
- `query`
- `context`
- `scenario_type`
- `intended_failure_mode`
- `query_type`
- `domain`

### ResponseRun

作用：

- 存储某个模型某次对某个 query 的回答

核心字段：

- `response_id`
- `query_id`
- `model_name`
- `run_id`
- `response_text`

### SentenceAnnotation

作用：

- 标注真实归因发生在哪个句子/片段
- 这是最可信的归因层

核心字段：

- `annotation_id`
- `response_id`
- `sentence_id`
- `sentence_text`
- `is_hallucinated`
- `attribution_type`
- `evidence_support`
- `severity`
- `notes`

---

## 3. query 层不再回答什么

query 层不再回答：

- “这道题就是真正的缺证断言题”
- “这道题就是真正的引入新事实题”

query 层只回答：

- 这道题设计上想诱发什么
- 它是不是一个好的诱发器
- 是否值得进入 response / sentence 标注池

---

## 4. 人工标注需要怎么改

### 旧的人审方式

旧方式里，人常被要求判断：

- `confirmed_target_error_type`
- `is_single_target_error`

这隐含着“query 本身有真实归因标签”的假设。

### 新的人审方式

新方式里，人应该主要判断：

- `confirmed_intended_failure_mode`
- `query_is_good_trigger`
- `query_is_natural`
- `verifier_design_is_feasible`
- `reward_should_be_verifiable`
- `requires_sentence_annotation`
- `sentence_annotation_priority`
- `decision`
- `notes`

### 这些字段分别是什么意思

`confirmed_intended_failure_mode`

- 人工确认这道题设计上想诱发什么
- 不是说这道题天然属于这个真实归因

`query_is_good_trigger`

- 这道题是否确实容易诱发目标错误
- 这是 query 级最重要的判断

`requires_sentence_annotation`

- 是否值得进入句子级真实归因标注池

`sentence_annotation_priority`

- `high / medium / low`

---

## 5. 最小人工标注模板

建议只保留下面 7 个核心字段：

- `reviewer`
- `decision`
- `confirmed_intended_failure_mode`
- `query_is_good_trigger`
- `verifier_design_is_feasible`
- `requires_sentence_annotation`
- `notes`

如果有余力，再补：

- `query_is_natural`
- `reward_should_be_verifiable`
- `sentence_annotation_priority`

---

## 6. release 决策怎么改

以后 release 不应再依赖：

- “query 是否有唯一真实归因”

而应依赖：

- `query_is_good_trigger`
- `verifier_design_is_feasible`
- `reward_should_be_verifiable`
- `requires_sentence_annotation`

也就是：

1. 它是不是一个好诱发器
2. 它能不能进入规则化评测
3. 它值不值得进入更贵的句子级标注

---

## 7. 推荐工作流

### Phase A

先做人审 query 层：

- intended failure mode 是否合理
- query 是否是好诱发器
- 是否进入 sentence 标注池

### Phase B

对通过筛选的 query，收集多个模型 response

### Phase C

只对高价值 response 做 sentence-level 真实归因标注

### Phase D

从 sentence 标注反推：

- 这道 query 是否真的是好的诱发器
- 哪种模型最容易在什么句子上出错
- 检测器是否漏掉了真实错误句子

---

## 8. 一句话总结

新的 schema 不是“query 对应一个真实归因标签”，而是：

`query 负责表达诱发目标，response 负责承载具体回答，sentence 负责承载真实归因。`
