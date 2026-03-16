# 人工标注数据规范

## review task JSONL

每行一个任务，核心字段：

- `item_id`
- `target_error_type`
- `scenario_type`
- `query`
- `context_preview`
- `reference_answer`
- `plan_id`
- `plan_summary`
- `auto_signals`
- `required_checks`
- `role_responsibilities`
- `review_result`

## review_result 规范

```json
{
  "reviewer": "alice",
  "decision": "approve",
  "confirmed_target_error_type": "错误拼接",
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
  "notes": "结构化状态清晰，可进入 verifier-based 训练数据池。"
}
```
