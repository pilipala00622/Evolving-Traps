# 人工标注规则

这份规则只服务于人工审核，不承担训练或自动验证逻辑。

## 审核目标

人工审核要回答 6 个问题：

1. 目标 `error-type` 是否主导
2. `reference_answer` 是否完全由 `context` 支撑
3. 是否能抽象出清晰的 `final_state`
4. 是否适合设计程序化 verifier
5. query / context 是否自然
6. 时间型题的时间元数据是否正确

## 审核结论定义

- `approve`
  可直接进入 verified data 池
- `revise`
  方向正确但仍需修正后再审
- `reject`
  不建议继续投入

## 必填布尔字段

- `reference_answer_supported`
- `final_state_is_correctly_specified`
- `verifier_design_is_feasible`
- `reward_should_be_verifiable`
- `query_is_natural`
- `is_single_target_error`

## issue_tags 建议枚举

- `mixed_error`
- `bad_reference`
- `weak_final_state`
- `verifier_hard_to_design`
- `unnatural_query`
- `time_mismatch`
- `paired_mismatch`
- `needs_more_context`
- `overfitted_prompt`

## 角色分工

- `error_reviewer`
  聚焦目标错误类型与混合错误
- `evidence_reviewer`
  聚焦 reference answer、final state、verifier feasibility
- `time_reviewer`
  聚焦 `real_time / out_of_date` 样本
- `release_reviewer`
  负责最终放行到 verified data 池
