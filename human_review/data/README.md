# human_review 数据目录

这个目录建议只存人工审核流程产物，不存自动生成的训练/评测中间结果。

## 子目录

- `tasks/`
  待审核任务
- `reviews/`
  审核员填写后的结果
- `releases/`
  merge 后的完整结果与 approve 子集

## 推荐文件名

- `tasks/benchmark_review_tasks.jsonl`
- `reviews/benchmark_review_tasks.reviewed.jsonl`
- `releases/benchmark_candidates.reviewed.json`
- `releases/benchmark_release_candidates.json`
