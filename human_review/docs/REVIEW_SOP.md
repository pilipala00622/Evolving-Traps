# 人工标注 SOP

## 目录约定

- `human_review/docs/`
  审核规则、流程、质检标准
- `human_review/templates/`
  HTML 审核页、JSONL 样例、字段模板
- `human_review/data/tasks/`
  导出的待审任务
- `human_review/data/reviews/`
  人工填写后的结果
- `human_review/data/releases/`
  merge 后的批准样本

## Step 1. 导出待审任务

```bash
python3 -m human_review.cli export \
  --input benchmark_candidates.json \
  --output human_review/data/tasks/benchmark_review_tasks.jsonl
```

## Step 2. 审核员填写

推荐两种方式：

- 直接编辑 JSONL
- 在本地打开 `human_review/templates/review_app.html` 辅助查看字段和规则

## Step 3. 回写

```bash
python3 -m human_review.cli merge \
  --input benchmark_candidates.json \
  --reviews human_review/data/reviews/benchmark_review_tasks.reviewed.jsonl \
  --output human_review/data/releases/benchmark_candidates.reviewed.json \
  --approved-output human_review/data/releases/benchmark_release_candidates.json
```

## Step 4. 喂给后续流程

- bench 只消费 `approved`
- SFT / RL 只消费同时满足：
  - `approved`
  - `reference_answer_supported=true`
  - `final_state_is_correctly_specified=true`
  - `verifier_design_is_feasible=true`
  - `reward_should_be_verifiable=true`
  - `is_single_target_error=true`
