"""
Compress per-question portraits into reusable prototype seeds for EvoHallu.

Pipeline:
1. Load question portraits produced by build_question_portraits.py
2. Keep high-value portraits that show real hallucination + clear attribution + discrimination
3. Group portraits into transferable seed prototypes
4. Export:
   - seed_prototypes_detailed.jsonl
   - seed_questions.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Dict, Iterable, List, Tuple

from build_question_portraits import read_jsonl, round4


DEFAULT_SOURCE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "第一阶段-14个模型-gpt51评测",
)


QUESTION_PORTRAITS_PATH = os.path.join(DEFAULT_SOURCE_DIR, "question_portraits_detailed.jsonl")


CANONICAL_DOMAINS = {
    "经济金融": "经济金融",
    "健康医疗": "健康医疗",
    "科技/互联网": "科技互联网",
    "科技互联网": "科技互联网",
    "教育考试": "教育考试",
    "法律司法": "法律政务",
    "法律政务": "法律政务",
    "政治时事": "法律政务",
    "传统行业": "传统行业",
    "工作职场": "工作职场",
    "自然科学": "自然科学",
    "文学艺术": "文化历史",
    "文化历史": "文化历史",
    "娱乐休闲": "娱乐休闲",
    "旅游摄影": "出行交通",
    "出行交通": "出行交通",
    "动物宠物": "动物宠物",
    "农林牧渔": "农林牧渔",
    "生活家庭": "家居生活",
    "家居生活": "家居生活",
    "情感心理": "家居生活",
    "餐饮美食": "家居生活",
    "军武军迷": "其他",
    "其他": "其他",
}


TASK_TYPES = ["信息定位", "边界感知", "文档整合", "生成控制"]
ATTRIBUTIONS = ["错误匹配", "引用错误", "限定错误", "缺证断言", "确定性膨胀", "过度概括", "引入新事实", "错误拼接"]
ALLOWED_LENGTHS = [5000, 15000, 30000, 50000, 80000, 120000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从题目画像表压缩生成 EvoHallu 种子原型")
    parser.add_argument("--portraits", default=QUESTION_PORTRAITS_PATH, help="question_portraits_detailed.jsonl 路径")
    parser.add_argument("--output-dir", default=None, help="输出目录，默认写回 portrait 文件所在目录")
    parser.add_argument("--min-avg-rate", type=float, default=0.03, help="最低平均幻觉率")
    parser.add_argument("--min-discrimination", type=float, default=0.05, help="最低区分度")
    parser.add_argument("--min-clarity", type=float, default=0.45, help="最低主导归因清晰度")
    parser.add_argument("--top-k-per-cluster", type=int, default=5, help="每个模式簇最多保留多少高分成员用于聚合")
    return parser.parse_args()


def canonicalize_domain(domain: str) -> str:
    return CANONICAL_DOMAINS.get(domain, "其他")


def nearest_allowed_length(value: float) -> int:
    return min(ALLOWED_LENGTHS, key=lambda x: abs(x - value))


def bucket_difficulty(value: float) -> str:
    if value < 0.35:
        return "low"
    if value < 0.65:
        return "mid"
    return "high"


def extract_attr_clarity(portrait: Dict) -> float:
    attr_dist = portrait.get("归因分布", {})
    total = sum(attr_dist.values())
    if not total:
        return 0.0
    top = max(attr_dist.values())
    return top / total


def compute_seed_score(portrait: Dict) -> float:
    clarity = extract_attr_clarity(portrait)
    score = 0.45 * portrait["区分度得分"] + 0.35 * portrait["平均幻觉率"] + 0.20 * clarity
    return round4(score)


def passes_filter(portrait: Dict, min_avg_rate: float, min_discrimination: float, min_clarity: float) -> bool:
    if portrait.get("主导归因类型") not in ATTRIBUTIONS:
        return False
    if portrait.get("能力板块") not in TASK_TYPES:
        return False
    if portrait.get("平均幻觉率", 0.0) < min_avg_rate:
        return False
    if portrait.get("区分度得分", 0.0) < min_discrimination:
        return False
    if extract_attr_clarity(portrait) < min_clarity:
        return False
    return True


def infer_context_defaults(attr: str, difficulty: float, doc_count: int, avg_question_length: float) -> Dict:
    semantic_similarity = 0.45
    shared_entities = 1
    distractor_ratio = min(0.6, 0.12 + difficulty * 0.4)
    answer_position = "mid"

    if attr in {"错误匹配", "限定错误"}:
        semantic_similarity = 0.72
        shared_entities = max(2, min(5, doc_count))
    elif attr == "错误拼接":
        semantic_similarity = 0.64
        shared_entities = max(2, min(6, doc_count + 1))
        distractor_ratio = min(0.7, 0.2 + difficulty * 0.45)
    elif attr in {"缺证断言", "过度概括", "引入新事实"}:
        semantic_similarity = 0.48
        shared_entities = max(1, min(3, doc_count))
    elif attr == "引用错误":
        semantic_similarity = 0.58
        shared_entities = max(1, min(4, doc_count))

    total_length = nearest_allowed_length(max(5000, avg_question_length))
    if total_length >= 50000:
        answer_position = "tail"

    return {
        "doc_count": max(1, min(15, doc_count)),
        "semantic_similarity": round4(semantic_similarity),
        "shared_entities": max(0, min(10, shared_entities)),
        "answer_position": answer_position,
        "distractor_ratio": round4(distractor_ratio),
        "total_length": total_length,
    }


def infer_trap_defaults(attr: str, difficulty: float, doc_count: int) -> Dict:
    confusion_pairs = 1
    evidence_clarity = max(0.25, 0.78 - difficulty * 0.35)
    hedging_level = 1
    info_gap = min(0.85, 0.18 + difficulty * 0.45)
    cross_doc_overlap = min(0.8, 0.15 + max(doc_count - 1, 0) * 0.08)

    if attr == "错误匹配":
        confusion_pairs = min(5, max(2, int(round(2 + difficulty * 3))))
        evidence_clarity = max(0.35, 0.7 - difficulty * 0.2)
        info_gap = min(0.5, 0.12 + difficulty * 0.25)
    elif attr == "限定错误":
        confusion_pairs = min(4, max(2, int(round(1 + difficulty * 3))))
        hedging_level = 1
        evidence_clarity = max(0.35, 0.62 - difficulty * 0.2)
    elif attr == "缺证断言":
        confusion_pairs = max(0, min(2, doc_count - 1))
        hedging_level = 1 if difficulty < 0.55 else 2
        evidence_clarity = max(0.25, 0.58 - difficulty * 0.25)
        info_gap = min(0.9, 0.3 + difficulty * 0.45)
    elif attr == "确定性膨胀":
        hedging_level = 2 if difficulty < 0.6 else 3
        evidence_clarity = max(0.2, 0.5 - difficulty * 0.2)
        info_gap = min(0.6, 0.15 + difficulty * 0.25)
    elif attr == "过度概括":
        hedging_level = 1
        evidence_clarity = max(0.28, 0.55 - difficulty * 0.2)
        info_gap = min(0.75, 0.22 + difficulty * 0.35)
    elif attr == "引入新事实":
        confusion_pairs = 0
        hedging_level = 1
        evidence_clarity = max(0.3, 0.6 - difficulty * 0.2)
        info_gap = min(0.95, 0.45 + difficulty * 0.35)
    elif attr == "错误拼接":
        confusion_pairs = max(1, min(3, doc_count - 1))
        evidence_clarity = max(0.28, 0.56 - difficulty * 0.18)
        info_gap = min(0.7, 0.18 + difficulty * 0.2)
        cross_doc_overlap = min(0.95, 0.42 + difficulty * 0.35)
    elif attr == "引用错误":
        confusion_pairs = max(1, min(3, doc_count - 1))
        evidence_clarity = max(0.3, 0.52 - difficulty * 0.18)
        info_gap = min(0.55, 0.12 + difficulty * 0.18)
        cross_doc_overlap = min(0.75, 0.24 + difficulty * 0.18)

    return {
        "confusion_pairs": max(0, min(5, confusion_pairs)),
        "evidence_clarity": round4(max(0.0, min(1.0, evidence_clarity))),
        "hedging_level": max(0, min(3, hedging_level)),
        "info_gap": round4(max(0.0, min(1.0, info_gap))),
        "cross_doc_overlap": round4(max(0.0, min(1.0, cross_doc_overlap))),
    }


def build_cluster_key(portrait: Dict) -> Tuple[str, str, str, str]:
    seed = portrait["seed_projection"]
    return (
        seed["task_type"],
        canonicalize_domain(seed["domain"]),
        seed["target_attribution"],
        bucket_difficulty(seed["target_difficulty"]),
    )


def aggregate_cluster(cluster_key: Tuple[str, str, str, str], members: List[Dict]) -> Dict:
    task_type, domain, target_attr, difficulty_bucket = cluster_key
    members = sorted(members, key=lambda x: (-x["seed_score"], x["prompt_id"]))

    target_difficulty = round4(mean(m["seed_projection"]["target_difficulty"] for m in members))
    complexity = int(round(median(m["seed_projection"]["complexity"] for m in members)))
    doc_count = int(round(median(m["seed_projection"]["doc_count"] for m in members)))
    avg_question_length = mean(len(m.get("问题", "")) for m in members)

    context = infer_context_defaults(target_attr, target_difficulty, doc_count, avg_question_length)
    trap = infer_trap_defaults(target_attr, target_difficulty, context["doc_count"])
    step_expansion = max(1, min(5, complexity + (1 if context["doc_count"] >= 3 else 0)))

    prototype = {
        "prototype_id": f"{task_type}-{domain}-{target_attr}-{difficulty_bucket}-{len(members)}",
        "cluster_key": {
            "task_type": task_type,
            "domain": domain,
            "target_attribution": target_attr,
            "difficulty_bucket": difficulty_bucket,
        },
        "support_count": len(members),
        "member_prompt_ids": [m["prompt_id"] for m in members],
        "member_tags": [m.get("tag") for m in members[:5]],
        "avg_seed_score": round4(mean(m["seed_score"] for m in members)),
        "avg_hallucination_rate": round4(mean(m["平均幻觉率"] for m in members)),
        "avg_discrimination": round4(mean(m["区分度得分"] for m in members)),
        "avg_attr_clarity": round4(mean(extract_attr_clarity(m) for m in members)),
        "raw_attr_counter": dict(Counter(m["主导归因类型"] for m in members).most_common()),
        "seed_question": {
            "query": {
                "task_type": task_type,
                "complexity": complexity,
            },
            "context": {
                "domain": domain,
                **context,
            },
            "trap": {
                **trap,
                "target_attribution": target_attr,
            },
            "difficulty": {
                "target_difficulty": target_difficulty,
                "step_expansion": step_expansion,
            },
        },
        "representative_examples": [
            {
                "prompt_id": m["prompt_id"],
                "tag": m.get("tag"),
                "avg_hallucination_rate": m["平均幻觉率"],
                "discrimination": m["区分度得分"],
                "attr_clarity": round4(extract_attr_clarity(m)),
            }
            for m in members[:3]
        ],
    }
    return prototype


def main() -> None:
    args = parse_args()
    portraits_path = os.path.abspath(args.portraits)
    output_dir = os.path.abspath(args.output_dir or os.path.dirname(portraits_path))
    os.makedirs(output_dir, exist_ok=True)

    portraits = read_jsonl(portraits_path)
    for portrait in portraits:
        portrait["seed_score"] = compute_seed_score(portrait)

    filtered = [
        p for p in portraits
        if passes_filter(
            p,
            min_avg_rate=args.min_avg_rate,
            min_discrimination=args.min_discrimination,
            min_clarity=args.min_clarity,
        )
    ]

    clusters: Dict[Tuple[str, str, str, str], List[Dict]] = defaultdict(list)
    for portrait in filtered:
        clusters[build_cluster_key(portrait)].append(portrait)

    prototypes = []
    for cluster_key, members in sorted(clusters.items()):
        top_members = sorted(members, key=lambda x: (-x["seed_score"], x["prompt_id"]))[: args.top_k_per_cluster]
        prototypes.append(aggregate_cluster(cluster_key, top_members))

    prototypes.sort(key=lambda x: (-x["avg_seed_score"], -x["support_count"], x["prototype_id"]))

    detailed_path = os.path.join(output_dir, "seed_prototypes_detailed.jsonl")
    seed_questions_path = os.path.join(output_dir, "seed_questions.json")

    with open(detailed_path, "w", encoding="utf-8") as f:
        for prototype in prototypes:
            f.write(json.dumps(prototype, ensure_ascii=False) + "\n")

    with open(seed_questions_path, "w", encoding="utf-8") as f:
        json.dump([p["seed_question"] for p in prototypes], f, ensure_ascii=False, indent=2)

    summary = {
        "portraits_path": portraits_path,
        "portrait_count": len(portraits),
        "filtered_count": len(filtered),
        "prototype_count": len(prototypes),
        "output_dir": output_dir,
        "seed_questions_path": seed_questions_path,
        "seed_prototypes_path": detailed_path,
        "attribution_distribution": dict(Counter(p["cluster_key"]["target_attribution"] for p in prototypes).most_common()),
        "task_distribution": dict(Counter(p["cluster_key"]["task_type"] for p in prototypes).most_common()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
