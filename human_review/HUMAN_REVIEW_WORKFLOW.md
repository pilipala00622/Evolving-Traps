# 人工检查与标注流程

这个流程单独负责把自动生成的 `benchmark_candidates.json` 变成可以正式入库、继续训练、继续评测的人工确认版本。

## 目标

人工审核不是重复跑模型，而是回答自动流程无法稳定保证的 4 类问题：

1. 这道题是否真的主要打在目标 `error-type`
2. `reference_answer` 是否完全由 `context` 支撑
3. 这道题是否有清晰、可验证的最终状态或 verifier 设计空间
4. 题目是否自然，像真实用户问题
5. 如果是 `real_time / out_of_date` 题，对应的时间元数据和配对关系是否正确

## 文件角色

- `benchmark_candidates.json`
  自动生成的候选集，字段完整，但不适合直接人工编辑
- `benchmark_review_tasks.jsonl`
  导出的待审任务，每行一题，给审核员填写
- `benchmark_candidates.reviewed.json`
  回写过人工审核结果的完整候选集
- `benchmark_release_candidates.json`
  只保留人工 `approve` 的题，可以继续进入 bench / 训练数据池

## 审核规则

### 1. 目标错误类型

需要人工确认：

- 该题的主要失败模式是否就是目标错误类型
- 是否存在“自动判成 A，但人工看更像 B”的情况
- 是否属于混合题

判定建议：

- `approve`
  主导错误类型明确，非目标错误只是次要噪声
- `revise`
  目标错误方向是对的，但题目混入了较多非目标错误，或者提示词/上下文还可修
- `reject`
  根本不是目标错误类型，或者问题设计太混乱，不值得修

### 2. 标准答案与证据支撑

必须人工确认：

- `reference_answer` 中每个关键信息点都能在 `context` 找到
- 没有引入常识补全、模型外部知识、时间外推

如果做不到，至少标记为 `revise`。

### 3. 可验证最终状态 / verifier 可行性

必须人工确认：

- 这道题能否抽象出结构化 `final_state`
- verifier 是否可以基于状态匹配、字段匹配或规则匹配判断成功
- 是否必须依赖模糊主观偏好才能判断“对/错”

如果只能靠模糊主观判断，不适合直接进入 verifier-based RL 数据池。

### 4. 题目自然性

重点检查：

- query 是否像真实用户会问的问题
- 是否出现明显“为了诱导某类错误而写得很假”的痕迹
- context 是否过度模板化

如果 query 很机械，但结构仍可修，建议 `revise`；如果整体过于伪造，建议 `reject`。

### 5. 时间型题 (`real_time / out_of_date`)

这部分必须人工确认：

- `knowledge_cutoff` 是否填写正确
- `context_timestamp` 是否填写正确
- paired item 是否只有“时间条件”不同，其他难度因子基本保持一致

如果 paired item 不对称，不能进入正式对照实验。

## 审核步骤

### Step 1. 导出待审任务

```bash
python3 -m human_review.cli export \
  --input benchmark_candidates.json \
  --output human_review/data/tasks/benchmark_review_tasks.jsonl
```

如果只想继续审未批准样本：

```bash
python3 -m human_review.cli export \
  --input human_review/data/releases/benchmark_candidates.reviewed.json \
  --output human_review/data/tasks/benchmark_review_tasks.pending.jsonl \
  --only-pending
```

### Step 2. 人工填写 `review_result`

每一行都有一个 `review_result`，审核员只需要填写这里：

```json
{
  "review_result": {
    "reviewer": "alice",
    "decision": "approve",
    "confirmed_target_error_type": "错误匹配",
    "confirmed_scenario_type": "static",
    "reference_answer_supported": true,
    "final_state_is_correctly_specified": true,
    "verifier_design_is_feasible": true,
    "reward_should_be_verifiable": true,
    "query_is_natural": true,
    "time_metadata_correct": null,
    "is_single_target_error": true,
    "release_priority": "high",
    "issue_tags": ["low_leakage"],
    "notes": "目标错误清晰，标答可由 context 支撑。"
  }
}
```

字段说明：

- `decision`
  只能填 `approve` / `revise` / `reject`
- `confirmed_target_error_type`
  人工最终确认的错误类型
- `confirmed_scenario_type`
  人工最终确认的场景类型
- `reference_answer_supported`
  标答是否完全由 context 支撑
- `final_state_is_correctly_specified`
  是否能明确写出任务完成后的结构化状态
- `verifier_design_is_feasible`
  是否能为该题设计程序化 verifier
- `reward_should_be_verifiable`
  该题是否适合作为后续 verifier reward 的样本
- `query_is_natural`
  是否像真实用户问题
- `time_metadata_correct`
  非时间题可留 `null`
- `is_single_target_error`
  是否主要只打一个目标错误
- `issue_tags`
  可自由补充，例如 `mixed_error`, `bad_reference`, `time_mismatch`
- `notes`
  记录修改建议、拒绝原因、可复用观察

## 回写和喂入

### Step 3. 回写人工结果

```bash
python3 -m human_review.cli merge \
  --input benchmark_candidates.json \
  --reviews human_review/data/reviews/benchmark_review_tasks.reviewed.jsonl \
  --output human_review/data/releases/benchmark_candidates.reviewed.json \
  --approved-output human_review/data/releases/benchmark_release_candidates.json
```

这一步会：

- 把人工结果写回 `human_review`
- 将 `approve` 样本单独导出到 `benchmark_release_candidates.json`

### Step 4. 喂给后续流程

`benchmark_release_candidates.json` 建议作为后续统一入口：

- 喂给 bench：
  只使用 `approve` 的题
- 喂给训练：
  优先使用 `approve` 且 `is_single_target_error=true`、`verifier_design_is_feasible=true` 的题
- 喂给迭代生成：
  把 `revise` / `reject` 的原因收集起来，作为下一轮 prototype 改进信号

## 哪些结论必须保留给后续系统

建议在后续训练/bench 流程里重点使用这些人工字段：

- `human_review.status`
- `human_review.labeled_target_error_type`
- `human_review.labeled_scenario_type`
- `human_review.review_result.reference_answer_supported`
- `human_review.review_result.final_state_is_correctly_specified`
- `human_review.review_result.verifier_design_is_feasible`
- `human_review.review_result.reward_should_be_verifiable`
- `human_review.review_result.query_is_natural`
- `human_review.review_result.time_metadata_correct`
- `human_review.review_result.is_single_target_error`
- `human_review.review_result.issue_tags`

其中最重要的是：

1. 人工最终确认的 `error-type`
2. 是否单目标错误
3. 是否能进入 verifier-based 训练数据池
4. 是否可作为正式发布题

## 推荐审核分工

- 标注员 A：看错误类型和 query 自然性
- 标注员 B：看 `reference_answer` 与 `context` 支撑关系
- 时间型专项审核员：只负责 `real_time / out_of_date` 元数据和 paired 对照关系

如果资源有限，至少保证：

- 每题有一人完整审核
- 时间型题额外二审
