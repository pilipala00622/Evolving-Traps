"""
Build per-question portrait tables from multi-model hallucination eval JSONL files.

Default source:
    第一阶段-14个模型-gpt51评测

Outputs:
    - question_portraits.csv
    - question_portraits_detailed.jsonl

Each portrait aggregates the same prompt across all aligned model result files.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List, Tuple


DEFAULT_SOURCE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "第一阶段-14个模型-gpt51评测",
)


NORMALIZED_ATTRIBUTIONS = [
    "错误匹配",
    "引用错误",
    "限定错误",
    "缺证断言",
    "确定性膨胀",
    "过度概括",
    "引入新事实",
    "错误拼接",
]


CSV_FIELDS = [
    "prompt_id",
    "一级分类",
    "二级分类",
    "三级分类",
    "tag",
    "能力板块",
    "题目预览",
    "模型数",
    "模型列表",
    "平均幻觉率",
    "最小幻觉率",
    "最大幻觉率",
    "幻觉率标准差",
    "幻觉率极差",
    "区分度得分",
    "平均错误句数",
    "平均需验证句数",
    "平均总句数",
    "主导原始错误类型",
    "主导归因类型",
    "归因分布",
    "高幻觉模型",
    "低幻觉模型",
    "建议_task_type",
    "建议_domain",
    "建议_target_attribution",
    "建议_target_difficulty",
    "建议_complexity",
    "估计_doc_count",
    "问题长度",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将多模型评测结果聚合为题目画像表")
    parser.add_argument(
        "--source-dir",
        default=DEFAULT_SOURCE_DIR,
        help="包含 eval_*.jsonl 的目录",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录，默认写回 source-dir",
    )
    parser.add_argument(
        "--include-non-common",
        action="store_true",
        help="是否保留并非所有模型都覆盖的题目",
    )
    return parser.parse_args()


def discover_eval_files(source_dir: str) -> List[str]:
    pattern = os.path.join(source_dir, "eval_*.jsonl")
    files = []
    for path in sorted(glob.glob(pattern)):
        if "processed" in os.path.basename(path):
            continue
        files.append(path)
    if not files:
        raise FileNotFoundError(f"在 {source_dir} 下未找到原始 eval_*.jsonl 文件")
    return files


def model_name_from_path(path: str) -> str:
    name = os.path.basename(path)
    name = re.sub(r"^eval_", "", name)
    name = re.sub(r"_step1_whole_.*\.jsonl$", "", name)
    return name


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_error_type(raw_error_type: str) -> List[str]:
    if not raw_error_type or raw_error_type == "无错误":
        return []

    result: List[str] = []

    def add(label: str) -> None:
        if label not in result:
            result.append(label)

    if "错误拼接" in raw_error_type:
        add("错误拼接")
    if "引用错误" in raw_error_type:
        add("引用错误")
    if "错误匹配" in raw_error_type:
        add("错误匹配")
    if "限定错误" in raw_error_type:
        add("限定错误")
    if "确定性膨胀" in raw_error_type:
        add("确定性膨胀")
    if "过度概括" in raw_error_type:
        add("过度概括")
    if (
        "引入新事实" in raw_error_type
        or "无中生有" in raw_error_type
        or "自行补充未要求信息" in raw_error_type
    ):
        add("引入新事实")
    if (
        "缺证断言" in raw_error_type
        or "越权推理" in raw_error_type
        or "结论外推" in raw_error_type
        or "因果越权" in raw_error_type
        or "隐含前提" in raw_error_type
    ):
        add("缺证断言")

    if not result and "生成错误" in raw_error_type:
        add("引入新事实")

    return result


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def safe_median(values: Iterable[float]) -> float:
    values = list(values)
    return median(values) if values else 0.0


def safe_pstdev(values: Iterable[float]) -> float:
    values = list(values)
    return pstdev(values) if len(values) > 1 else 0.0


def round4(value: float) -> float:
    return round(float(value), 4)


def compute_discrimination_score(rates: List[float]) -> float:
    if len(rates) < 2:
        return 0.0
    avg_rate = sum(rates) / len(rates)
    variance = sum((x - avg_rate) ** 2 for x in rates) / len(rates)
    normalized = min(variance / 0.25, 1.0)
    spread = max(rates) - min(rates)
    spread_bonus = max(0.0, (spread - 0.3) * 0.5)
    return round4(min(normalized + spread_bonus, 1.0))


def estimate_complexity(ability: str, third_category: str, question: str) -> int:
    if ability == "文档整合":
        return 3
    if ability == "边界感知":
        return 2
    if ability == "信息定位":
        return 1
    if ability == "生成控制":
        return 3

    if third_category == "推理":
        return 3
    if third_category == "生成":
        return 3
    if "结合" in question or "综合" in question:
        return 3
    return 1


def estimate_doc_count(question: str) -> int:
    patterns = [
        question.count("标题: "),
        question.count("##《"),
        question.count("<标题>"),
        question.count("相关文档名："),
    ]
    count = max(patterns)
    return max(1, min(count if count > 0 else 1, 15))


def estimate_target_difficulty(avg_rate: float, discrimination: float) -> float:
    score = 0.55 * avg_rate + 0.45 * discrimination
    return round4(min(max(score, 0.0), 1.0))


def preview_text(text: str, limit: int = 140) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def top_items(counter: Counter, n: int = 3) -> List[Tuple[str, int]]:
    return counter.most_common(n)


def build_prompt_index(
    files: List[str], include_non_common: bool = False
) -> Tuple[Dict[str, Dict[str, Dict]], List[str]]:
    model_to_rows: Dict[str, Dict[str, Dict]] = {}
    prompt_sets: List[set] = []

    for path in files:
        model = model_name_from_path(path)
        rows = read_jsonl(path)
        indexed = {}
        for row in rows:
            prompt_id = row.get("Prompt序列号")
            if prompt_id:
                indexed[prompt_id] = row
        model_to_rows[model] = indexed
        prompt_sets.append(set(indexed.keys()))

    if include_non_common:
        prompt_ids = sorted(set().union(*prompt_sets))
    else:
        common = set.intersection(*prompt_sets)
        prompt_ids = sorted(common)

    prompt_index: Dict[str, Dict[str, Dict]] = {}
    for prompt_id in prompt_ids:
        prompt_index[prompt_id] = {}
        for model, indexed in model_to_rows.items():
            row = indexed.get(prompt_id)
            if row is not None:
                prompt_index[prompt_id][model] = row

    return prompt_index, sorted(model_to_rows.keys())


def aggregate_portrait(prompt_id: str, model_rows: Dict[str, Dict], all_models: List[str]) -> Dict:
    sample = next(iter(model_rows.values()))
    question = sample.get("问题", "")
    reference_answer = sample.get("参考答案")
    ability = sample.get("能力板块") or sample.get("三级分类") or ""
    third_category = sample.get("三级分类") or ""

    rates: List[float] = []
    error_counts: List[float] = []
    verify_counts: List[float] = []
    sentence_counts: List[float] = []
    raw_error_counter: Counter = Counter()
    normalized_attr_counter: Counter = Counter()
    per_model = []

    for model in all_models:
        row = model_rows.get(model)
        if row is None:
            continue

        rate = float(row.get("hallucination_rate_record") or 0.0)
        error_count = int(row.get("error_count") or 0)
        verify_count = int(row.get("need_verify_count") or 0)
        total_sentences = int(row.get("total_sentences") or len(row.get("sentence_results") or []))
        model_error_counter: Counter = Counter()

        for sentence_result in row.get("sentence_results") or []:
            result = sentence_result.get("result") or {}
            error = result.get("error") or {}
            raw_type = error.get("error_type")
            if raw_type:
                raw_error_counter[raw_type] += 1
                model_error_counter[raw_type] += 1
                for normalized in normalize_error_type(raw_type):
                    normalized_attr_counter[normalized] += 1

        rates.append(rate)
        error_counts.append(error_count)
        verify_counts.append(verify_count)
        sentence_counts.append(total_sentences)
        per_model.append(
            {
                "model": model,
                "hallucination_rate": round4(rate),
                "error_count": error_count,
                "need_verify_count": verify_count,
                "total_sentences": total_sentences,
                "top_raw_error_types": top_items(model_error_counter, 3),
            }
        )

    discrimination = compute_discrimination_score(rates)
    avg_rate = safe_mean(rates)
    min_rate = min(rates) if rates else 0.0
    max_rate = max(rates) if rates else 0.0
    std_rate = safe_pstdev(rates)
    spread_rate = max_rate - min_rate if rates else 0.0

    dominant_raw_error_type = raw_error_counter.most_common(1)[0][0] if raw_error_counter else "无错误"
    dominant_attribution = (
        normalized_attr_counter.most_common(1)[0][0] if normalized_attr_counter else "无明显主导归因"
    )
    difficulty = estimate_target_difficulty(avg_rate, discrimination)
    complexity = estimate_complexity(ability, third_category, question)
    doc_count = estimate_doc_count(question)

    sorted_models_by_rate = sorted(per_model, key=lambda x: x["hallucination_rate"], reverse=True)

    portrait = {
        "prompt_id": prompt_id,
        "一级分类": sample.get("一级分类"),
        "二级分类": sample.get("二级分类"),
        "三级分类": third_category,
        "tag": sample.get("tag"),
        "能力板块": ability,
        "问题": question,
        "参考答案": reference_answer,
        "模型数": len(per_model),
        "模型列表": [item["model"] for item in per_model],
        "平均幻觉率": round4(avg_rate),
        "最小幻觉率": round4(min_rate),
        "最大幻觉率": round4(max_rate),
        "幻觉率标准差": round4(std_rate),
        "幻觉率极差": round4(spread_rate),
        "区分度得分": discrimination,
        "平均错误句数": round4(safe_mean(error_counts)),
        "平均需验证句数": round4(safe_mean(verify_counts)),
        "平均总句数": round4(safe_mean(sentence_counts)),
        "主导原始错误类型": dominant_raw_error_type,
        "主导归因类型": dominant_attribution,
        "原始错误分布": dict(raw_error_counter.most_common()),
        "归因分布": {k: normalized_attr_counter.get(k, 0) for k in NORMALIZED_ATTRIBUTIONS if normalized_attr_counter.get(k, 0) > 0},
        "高幻觉模型": [item["model"] for item in sorted_models_by_rate[:3]],
        "低幻觉模型": [item["model"] for item in sorted_models_by_rate[-3:]],
        "per_model": per_model,
        "seed_projection": {
            "task_type": ability,
            "domain": sample.get("二级分类"),
            "target_attribution": dominant_attribution,
            "target_difficulty": difficulty,
            "complexity": complexity,
            "doc_count": doc_count,
        },
    }
    return portrait


def write_csv(path: str, portraits: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in portraits:
            writer.writerow(
                {
                    "prompt_id": item["prompt_id"],
                    "一级分类": item["一级分类"],
                    "二级分类": item["二级分类"],
                    "三级分类": item["三级分类"],
                    "tag": item["tag"],
                    "能力板块": item["能力板块"],
                    "题目预览": preview_text(item["问题"]),
                    "模型数": item["模型数"],
                    "模型列表": json.dumps(item["模型列表"], ensure_ascii=False),
                    "平均幻觉率": item["平均幻觉率"],
                    "最小幻觉率": item["最小幻觉率"],
                    "最大幻觉率": item["最大幻觉率"],
                    "幻觉率标准差": item["幻觉率标准差"],
                    "幻觉率极差": item["幻觉率极差"],
                    "区分度得分": item["区分度得分"],
                    "平均错误句数": item["平均错误句数"],
                    "平均需验证句数": item["平均需验证句数"],
                    "平均总句数": item["平均总句数"],
                    "主导原始错误类型": item["主导原始错误类型"],
                    "主导归因类型": item["主导归因类型"],
                    "归因分布": json.dumps(item["归因分布"], ensure_ascii=False),
                    "高幻觉模型": json.dumps(item["高幻觉模型"], ensure_ascii=False),
                    "低幻觉模型": json.dumps(item["低幻觉模型"], ensure_ascii=False),
                    "建议_task_type": item["seed_projection"]["task_type"],
                    "建议_domain": item["seed_projection"]["domain"],
                    "建议_target_attribution": item["seed_projection"]["target_attribution"],
                    "建议_target_difficulty": item["seed_projection"]["target_difficulty"],
                    "建议_complexity": item["seed_projection"]["complexity"],
                    "估计_doc_count": item["seed_projection"]["doc_count"],
                    "问题长度": len(item["问题"]),
                }
            )


def write_jsonl(path: str, portraits: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in portraits:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    source_dir = os.path.abspath(args.source_dir)
    output_dir = os.path.abspath(args.output_dir or source_dir)
    os.makedirs(output_dir, exist_ok=True)

    eval_files = discover_eval_files(source_dir)
    prompt_index, all_models = build_prompt_index(
        eval_files,
        include_non_common=args.include_non_common,
    )

    portraits = [
        aggregate_portrait(prompt_id, model_rows, all_models)
        for prompt_id, model_rows in prompt_index.items()
    ]

    portraits.sort(
        key=lambda x: (
            -x["区分度得分"],
            -x["平均幻觉率"],
            x["prompt_id"],
        )
    )

    csv_path = os.path.join(output_dir, "question_portraits.csv")
    jsonl_path = os.path.join(output_dir, "question_portraits_detailed.jsonl")

    write_csv(csv_path, portraits)
    write_jsonl(jsonl_path, portraits)

    summary = {
        "source_dir": source_dir,
        "output_dir": output_dir,
        "model_count": len(all_models),
        "portrait_count": len(portraits),
        "models": all_models,
        "csv_path": csv_path,
        "jsonl_path": jsonl_path,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
