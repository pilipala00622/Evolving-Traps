# Response / Sentence 标注流程

## 1. 目标

这条流程不再要求人给 `query` 直接贴真实归因标签。

它做的是：

1. 从真实模型回答里导出 `response_runs`
2. 将回答切成句子，生成 `sentence_annotation_tasks`
3. 人工在句子级标出真实幻觉和归因
4. 自动汇总回 `response-level` 和 `query-level`

---

## 2. 先准备待标注任务

如果你已经有 query 层 review 结果，可以拿它做过滤：

```bash
python3 response_sentence_annotation_flow.py prepare \
  --query-review-jsonl path/to/reviewed_query.jsonl
```

如果只是先试跑：

```bash
python3 response_sentence_annotation_flow.py prepare
```

默认会生成：

- `demo_outputs/response_sentence_annotations/response_runs.jsonl`
- `demo_outputs/response_sentence_annotations/sentence_annotation_tasks.jsonl`
- `demo_outputs/response_sentence_annotations/sentence_annotation_tasks.xlsx`

---

## 3. 人工标注哪些字段

在 `sentence_annotation_tasks.xlsx` 里主要填写：

- `annotator`
- `is_hallucinated`
- `attribution_type`
- `evidence_support`
- `severity`
- `notes`

### 含义

`is_hallucinated`

- 这句是否存在幻觉

`attribution_type`

- 如果这句是幻觉，属于什么归因类型

`evidence_support`

- 简短写明为什么判断它有/没有证据支撑

`severity`

- `low / medium / high`

---

## 4. 标完后回收

先把 Excel 转回 JSONL：

```bash
python3 response_sentence_excel_bridge.py import \
  --input demo_outputs/response_sentence_annotations/sentence_annotation_tasks.xlsx \
  --output demo_outputs/response_sentence_annotations/sentence_annotation_tasks.reviewed.jsonl
```

然后汇总：

```bash
python3 response_sentence_annotation_flow.py summarize \
  --input demo_outputs/response_sentence_annotations/sentence_annotation_tasks.reviewed.jsonl
```

会得到：

- `sentence_annotations.jsonl`
- `response_summaries.json`
- `query_summaries.json`

---

## 5. 最关键的原则

### query 层

只表达：

- intended failure mode
- 是否是好的诱发器

### sentence 层

才表达：

- 这句是否真的幻觉
- 它属于什么归因

这才是后续检测器评估最可信的依据。
