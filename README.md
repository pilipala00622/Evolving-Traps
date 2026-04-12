# EvoHallu / GRIT / HalluSEA

面向企业噪声 RAG 场景的结构化幻觉诱发、评测、筛选与训练信号生成框架。

当前仓库已经切到新的主线语义：

- `query` 不是“真实归因标签”，而是 `intended trigger`
- `response / sentence` 才承载真实 failure attribution
- trap 不再只是一层粗标签，而是 `mechanism + manifestation + structure axes + difficulty`

## 先回答两个关键问题

### 1. 已有 `540` 条数据，下一步是不是先拿不同模型回答？

是。当前最合理的第一步就是先对 [data/幻觉评测集_v6_540条.jsonl](/Users/xyx/VscodeProjects/EvalBest/xyx_eval_diary/Eval_everything_in_my_era/hallucinate_eval_all/new_eval_for_IMA/attribution_agent/幻觉评测系统_v1.1/evo_hallucination/data/幻觉评测集_v6_540条.jsonl) 做多模型回答采集。

原因很简单：

- 你需要先看到不同模型在同一题上的真实输出分布
- 后面的 sentence 标注、keep/drop、latent gene、GRPO reward 都依赖真实回答
- 没有真实回答，`query` 只能停留在诱发意图，不能验证它是不是好 trap

### 2. 要不要保留 COT？

建议是：

- `必须保留`：最终答案、原始可见输出、trace/usage/timing
- `可选保留`：模型显式吐出来的可见 reasoning
- `不要依赖`：hidden / provider-internal CoT

也就是说，这个仓库现在默认采用：

- 评测主对象是 `final_answer`
- 如果模型真的把思考过程显式输出了，可以单独存到 `visible_reasoning`
- 不把隐藏 CoT 当成监督标签、评测金标准或 reward 依据

原因是 hidden CoT 通常：

- 不稳定
- 不同供应商不可比
- 训练和评测时往往拿不到
- 很容易让 benchmark 目标漂移到“学会模仿思维文本”而不是“学会守住证据边界”

## 当前 trap schema v2

核心结构在 [core/round_manager.py](/Users/xyx/VscodeProjects/EvalBest/xyx_eval_diary/Eval_everything_in_my_era/hallucinate_eval_all/new_eval_for_IMA/attribution_agent/幻觉评测系统_v1.1/evo_hallucination/core/round_manager.py)。

每个 gene 现在至少有这些关键轴：

- `failure_mechanism`
- `manifestation_hint`
- `answer_carrier`
- `evidence_layout`
- `pressure_pattern`
- `distractor_style`
- `boundary_scope`
- `difficulty`

`difficulty` 不是自由文本，而是结构化字段：

- `gap_concealment`
- `distractor_density`
- `composition_depth`
- `pressure_intensity`
- `verification_complexity`
- `score`
- `bucket`

## 仓库结构

现在根目录只保留包目录、文档和本地依赖。所有流程脚本都收进了子目录。

```text
evo_hallucination/
├── README.md
├── DESIGN_NOTES.md
├── EXPERIMENT_GUIDE.md
├── core/
│   ├── __init__.py
│   ├── agent_specs.py
│   ├── round_manager.py
│   ├── spec_factory.py
│   └── training_readiness.py
├── hallusea/
│   ├── __init__.py
│   ├── converter.py
│   └── curriculum.py
├── grpo/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset_builder.py
│   ├── ms_swift_plugin.py
│   ├── reward.py
│   └── train.py
├── pipelines/
│   ├── genes/
│   │   ├── extract_seed_genes.py
│   │   ├── normalize_gene_bank.py
│   │   ├── merge_gene_archive.py
│   │   └── run_gene_evolution.py
│   ├── generation/
│   │   ├── expand_genes_to_candidates.py
│   │   └── induce_from_source_contexts.py
│   ├── eval/
│   │   ├── collect_model_answers.py
│   │   └── evaluate_hard_hallucination_candidates.py
│   ├── analysis/
│   │   └── latent_gene_analysis.py
│   ├── benchmarks/
│   │   ├── build_benchmark_slices.py
│   │   └── package_benchmark_release.py
│   └── orchestration/
│       └── orchestrator.py
├── data/
├── reference/
└── llm.py
```

## 推荐的当前执行顺序

### Step 0：先采多模型回答

用 [pipelines/eval/collect_model_answers.py](/Users/xyx/VscodeProjects/EvalBest/xyx_eval_diary/Eval_everything_in_my_era/hallucinate_eval_all/new_eval_for_IMA/attribution_agent/幻觉评测系统_v1.1/evo_hallucination/pipelines/eval/collect_model_answers.py)：

```bash
python3 -m pipelines.eval.collect_model_answers \
  --input data/幻觉评测集_v6_540条.jsonl \
  --output runs/v6_540/model_answers.jsonl \
  --models qwen3.6-plus deepseek-v3.2 hunyuan-2.0-thinking-20251109 gpt-5.4 \
  --max-workers 4 \
  --preserve-visible-cot
```

输出每条记录会保留：

- `response_text`
- `final_answer`
- `visible_reasoning`
- `reasoning_capture_mode`
- `trace_id`
- `usage_info`
- `timing_info`

注意：

- `hidden_cot_preserved` 永远是 `false`
- 这个脚本不会尝试拿 provider 内部思维链

### Step 1：如果你已有人工种子，先抽 gene

```bash
python3 -m pipelines.genes.extract_seed_genes \
  --input data/reviewed_seeds.jsonl \
  --output runs/genes/seed_genes.raw.jsonl
```

然后标准化：

```bash
python3 -m pipelines.genes.normalize_gene_bank \
  --input runs/genes/seed_genes.raw.jsonl \
  --output runs/genes/seed_genes.norm.jsonl
```

### Step 2：gene 扩题

```bash
python3 -m pipelines.generation.expand_genes_to_candidates \
  --seeds data/reviewed_seeds.jsonl \
  --genes runs/genes/seed_genes.norm.jsonl \
  --contexts data/source_contexts.jsonl \
  --output runs/candidates/candidates.jsonl
```

或者做 context transfer：

```bash
python3 -m pipelines.generation.induce_from_source_contexts \
  --manifest runs/manifests/induction_manifest.jsonl \
  --contexts data/source_contexts.jsonl \
  --output runs/candidates/induction_results.jsonl
```

### Step 3：自动评测

```bash
python3 -m pipelines.eval.evaluate_hard_hallucination_candidates \
  --candidates runs/candidates/candidates.jsonl \
  --models qwen3.6-plus deepseek-v3.2 hunyuan-2.0-thinking-20251109 gpt-5.4 \
  --output-dir runs/eval \
  --round-id 0 \
  --model-version baseline_v0
```

### Step 4：基因进化

构建 population：

```bash
python3 -m pipelines.genes.run_gene_evolution build-population \
  --genes runs/genes/seed_genes.norm.jsonl \
  --candidates runs/candidates/candidates.jsonl \
  --eval-results runs/eval/model_answers_and_autoeval.jsonl \
  --output runs/genes/population.jsonl \
  --generation 0 \
  --round-id 0 \
  --model-version baseline_v0
```

变异：

```bash
python3 -m pipelines.genes.run_gene_evolution mutate \
  --population runs/genes/population.jsonl \
  --output runs/genes/mutated_children.jsonl \
  --round-id 0 \
  --model-version baseline_v0
```

### Step 5：benchmark 切片与发布

```bash
python3 -m pipelines.benchmarks.build_benchmark_slices \
  --out-dir runs/benchmark_slices

python3 -m pipelines.benchmarks.package_benchmark_release \
  --base runs/benchmark_slices \
  --out-dir runs/release_bundle
```

### Step 6：latent gene 分析

```bash
python3 -m pipelines.analysis.latent_gene_analysis \
  --query-decisions runs/query_keep_drop.json \
  --sentence-annotations runs/sentence_annotations.jsonl \
  --output-dir runs/latent_gene
```

### Step 7：生成 ms-swift 版 Qwen2.5-7B GRPO 训练包

```bash
python3 grpo/train.py \
  --hallusea-dir runs/hallusea_r0 \
  --output-dir runs/grpo_qwen25_7b \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --swift-bin /path/to/ms-swift/swift \
  --dry-run
```

这一步目前会生成：

- `ms_swift_grpo_config.json`
- `ms_swift_grpo_dataset.jsonl`
- `ms_swift_grpo_dataset_manifest.json`
- `ms_swift_launch_manifest.json`
- `run_ms_swift_grpo.sh`

也就是 `ms-swift` 训练前的 dataset / reward plugin / launch bundle。

注意：

- 现在这套 GRPO 已经按 `ms-swift` 原生方式准备，而不是旧的 TRL / OpenRLHF 通用 bundle
- `grpo/ms_swift_plugin.py` 会复用仓库现有的 `grpo/reward.py` 判分逻辑
- macOS 上系统自带的 `/usr/bin/swift` 是 Apple Swift 编译器，不是 `ms-swift`
- 所以实际训练时，建议显式传 `--swift-bin` 指向你安装好的 `ms-swift` launcher，或者运行生成出来的 `run_ms_swift_grpo.sh` 时覆写 `SWIFT_RLHF_BIN`

## 现阶段的评测原则

1. 先拿真实多模型回答，再谈 trap 优劣。
2. `query` 是诱发器，不是真实归因标签。
3. 真正的 failure attribution 应该尽量下沉到 `response / sentence`。
4. `difficulty` 分成设计难度和后验难度，不要只看 TEHR。
5. reward 要和 evaluator 同源，不要训练一套、评测另一套。

## 当前最推荐的下一步

如果你现在手里已经有 [data/幻觉评测集_v6_540条.jsonl](/Users/xyx/VscodeProjects/EvalBest/xyx_eval_diary/Eval_everything_in_my_era/hallucinate_eval_all/new_eval_for_IMA/attribution_agent/幻觉评测系统_v1.1/evo_hallucination/data/幻觉评测集_v6_540条.jsonl)，建议立刻做这两步：

1. 用 `collect_model_answers` 跑 4 到 6 个代表模型，先拿一版真实回答池。
2. 从里面抽一小包做 sentence-level 标注，验证哪些 query 真的是好诱发器。

这会比直接扩大 gene 或直接开始 GRPO 更稳，因为它先把“真实错误证据”补齐了。
