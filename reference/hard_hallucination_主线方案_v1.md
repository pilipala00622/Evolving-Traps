# Hard Hallucination 主线方案 v1

从现在开始，仓库主线只围绕下面 3 类 hard hallucination：

1. `弱证据强结论`
2. `缺失关键信息下硬答`
3. `背景证据当直接证据`

这条主线的核心不是 verifier pass rate，而是：

- 能不能稳定设计出真正诱发 hallucination 的题
- 能不能让题面清楚、边界危险
- 能不能让模型的正确行为是克制，而不是普通抽取

## 工作流

1. 从原始知识库数据抽纯文档 context
2. 基于 context 出三类 hard hallucination 题
3. 人工审核这些题是否真的构成 hallucination trigger
4. 再进入跨模型稳定性实验

## 为什么抛弃旧主线

旧主线的问题是：
- 太多题本质上是结构化抽取难题
- pass/fail 能跑，但不一定真的揭示幻觉
- 难度提升了，不等于 hallucination induction 提升了

新主线要求：
- 正确答案应该体现“不要越界”
- 错误答案应该明显体现 unsupported claim
- 评测重点从“答对没有”转到“是否诱发目标幻觉”

## 研究主张

如果新主线成立，需要证明：

1. 这三类题能稳定诱发目标幻觉，而不是普通错误
2. 这些题对不同模型仍有区分度
3. 同一题在多模型上能较稳定复现目标错误

## 主指标

- `target_trigger_rate`
- `non_target_leakage`
- `precision_of_target_error`
- `judgeable_rate`
- `cross_model_stability`

pass rate 只能作为辅助指标，不再是主结果。
