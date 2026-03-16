# 最小 demo 人工复核流程

## 1. 先确保 demo 已经跑过

```bash
python3 run_minimal_demo.py
```

## 2. 生成待标注 review 包

```bash
python3 minimal_demo_review_flow.py prepare
```

生成文件：

- `demo_outputs/minimal_closed_loop_demo/human_review_tasks.pending.jsonl`

## 3. 直接填写 review_result

你只需要在每一行的 `review_result` 里填写这些字段：

- `reviewer`
- `decision`
- `reference_answer_supported`
- `final_state_is_correctly_specified`
- `verifier_design_is_feasible`
- `reward_should_be_verifiable`
- `query_is_natural`
- `is_single_target_error`
- `notes`

`decision` 只能填：

- `approve`
- `revise`
- `reject`

文件里已经带了：

- `auto_signals`
- `auto_recommendation`

它们只是辅助参考，不会替你做最终判断。

## 4. 标完后继续跑后半段分析

如果你直接在默认文件里标：

```bash
python3 minimal_demo_review_flow.py finalize
```

如果你另存了一个 reviewed 文件：

```bash
python3 minimal_demo_review_flow.py finalize \
  --reviews path/to/your_reviewed.jsonl
```

## 5. 标完后会自动产出

- `benchmark_candidates.reviewed.human.json`
- `benchmark_release_candidates.human.json`
- `rollout_results.human.jsonl`
- `rollout_summary.human.json`
- `benchmark_subset_v1.human.json`
- `failure_pattern_analysis.human.json`
- `taxonomy_v2_human.md`
- `demo_report.human.md`

## 6. 我下一步会重点看什么

你标完后，我会重点看：

1. `approve` 的题是否真的形成了可验证样本池
2. 人工判断和自动信号冲突最大的地方在哪里
3. rollout 和 verifier 在人工筛选后是否更稳定
4. GA 选出的 stable subset 是否比原始 release 集更可靠
5. taxonomy v2 的新增模式是不是有真实必要
