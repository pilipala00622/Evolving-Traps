"""Utilities for exporting and summarizing response/sentence annotations."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

from core.annotation_schema import (
    QueryEvaluationSummary,
    QueryItem,
    ResponseEvaluationSummary,
    ResponseRun,
    SentenceAnnotation,
)


SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])")


def model_name_from_eval_filename(path: Path) -> str:
    name = path.name
    if name.startswith("eval_"):
        name = name[len("eval_") :]
    if "_step1_whole_" in name:
        name = name.split("_step1_whole_")[0]
    return name


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    if parts:
        return parts
    return [text]


def normalize_attribution_type(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw or raw == "无错误":
        return ""
    return raw


def build_sentence_annotation_tasks_from_eval_row(
    *,
    eval_row: Dict[str, object],
    model_name: str,
    intended_failure_mode: str,
) -> Tuple[ResponseRun, List[Dict[str, object]]]:
    query_id = str(eval_row.get("Prompt序列号", ""))
    response_id = f"{query_id}__{model_name}"
    response_text = str(eval_row.get("模型回答", "") or "")

    response_run = ResponseRun(
        response_id=response_id,
        query_id=query_id,
        model_name=model_name,
        run_id=model_name,
        response_text=response_text,
        metadata={
            "一级分类": eval_row.get("一级分类"),
            "二级分类": eval_row.get("二级分类"),
            "三级分类": eval_row.get("三级分类"),
            "能力板块": eval_row.get("能力板块"),
            "intended_failure_mode": intended_failure_mode,
        },
    )

    tasks: List[Dict[str, object]] = []
    sentence_results = list(eval_row.get("sentence_results") or [])
    if sentence_results:
        iterable = []
        for sr in sentence_results:
            result = sr.get("result") or {}
            error = result.get("error") or {}
            iterable.append(
                {
                    "sentence_index": sr.get("sentence_index"),
                    "sentence_text": sr.get("sentence", ""),
                    "prefill_is_hallucinated": result.get("is_hallucinated"),
                    "prefill_attribution_type": normalize_attribution_type(error.get("error_type", "")),
                    "prefill_evidence_support": result.get("evidence", ""),
                }
            )
    else:
        iterable = [
            {
                "sentence_index": idx,
                "sentence_text": sentence,
                "prefill_is_hallucinated": None,
                "prefill_attribution_type": "",
                "prefill_evidence_support": "",
            }
            for idx, sentence in enumerate(split_sentences(response_text), start=1)
        ]

    for entry in iterable:
        sentence_id = f"{response_id}__sent_{entry['sentence_index']}"
        tasks.append(
            {
                "annotation_id": sentence_id,
                "response_id": response_id,
                "query_id": query_id,
                "model_name": model_name,
                "intended_failure_mode": intended_failure_mode,
                "query": eval_row.get("问题", ""),
                "sentence_id": sentence_id,
                "sentence_index": entry["sentence_index"],
                "sentence_text": entry["sentence_text"],
                "prefill_is_hallucinated": entry["prefill_is_hallucinated"],
                "prefill_attribution_type": entry["prefill_attribution_type"],
                "prefill_evidence_support": entry["prefill_evidence_support"],
                "annotation_result": {
                    "annotator": "",
                    "is_hallucinated": None,
                    "attribution_type": "",
                    "evidence_support": "",
                    "severity": "",
                    "notes": "",
                },
            }
        )

    return response_run, tasks


def load_reviewed_query_filter(path: Optional[Path]) -> Dict[str, Dict[str, object]]:
    if path is None or not path.exists():
        return {}
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    result: Dict[str, Dict[str, object]] = {}
    for row in rows:
        rr = row.get("review_result", {})
        query_id = str(row.get("item_id", ""))
        if not query_id:
            continue
        if rr.get("requires_sentence_annotation") is False:
            continue
        if rr.get("decision") and rr.get("decision") == "reject":
            continue
        result[query_id] = row
    return result


def summarize_annotations(
    *,
    tasks: List[Dict[str, object]],
) -> Tuple[List[SentenceAnnotation], List[ResponseEvaluationSummary], List[QueryEvaluationSummary]]:
    sentence_annotations: List[SentenceAnnotation] = []
    by_response: Dict[str, List[SentenceAnnotation]] = defaultdict(list)
    intended_mode_by_query: Dict[str, str] = {}

    for task in tasks:
        result = task.get("annotation_result", {})
        if result.get("is_hallucinated", None) is None:
            continue
        annotation = SentenceAnnotation(
            annotation_id=str(task.get("annotation_id", "")),
            response_id=str(task.get("response_id", "")),
            sentence_id=str(task.get("sentence_id", "")),
            sentence_text=str(task.get("sentence_text", "")),
            is_hallucinated=bool(result.get("is_hallucinated", False)),
            attribution_type=str(result.get("attribution_type", "")),
            evidence_support=str(result.get("evidence_support", "")),
            severity=str(result.get("severity", "")),
            notes=str(result.get("notes", "")),
        )
        sentence_annotations.append(annotation)
        by_response[annotation.response_id].append(annotation)
        intended_mode_by_query[str(task.get("query_id", ""))] = str(task.get("intended_failure_mode", ""))

    response_summaries: List[ResponseEvaluationSummary] = []
    by_query_response_summary: Dict[str, List[ResponseEvaluationSummary]] = defaultdict(list)

    for response_id, annotations in by_response.items():
        counts = Counter(
            ann.attribution_type for ann in annotations if ann.is_hallucinated and ann.attribution_type
        )
        dominant = counts.most_common(1)[0][0] if counts else ""
        query_id = response_id.split("__")[0]
        summary = ResponseEvaluationSummary(
            response_id=response_id,
            dominant_attribution_type=dominant,
            hallucinated_sentence_count=sum(1 for ann in annotations if ann.is_hallucinated),
            total_sentence_count=len(annotations),
            attribution_distribution=dict(counts),
            metadata={"query_id": query_id},
        )
        response_summaries.append(summary)
        by_query_response_summary[query_id].append(summary)

    query_summaries: List[QueryEvaluationSummary] = []
    for query_id, summaries in by_query_response_summary.items():
        intended = intended_mode_by_query.get(query_id, "")
        dominant_counts = Counter(summary.dominant_attribution_type for summary in summaries if summary.dominant_attribution_type)
        dominant_observed = dominant_counts.most_common(1)[0][0] if dominant_counts else ""
        hit_rate = 0.0
        if summaries:
            hit_rate = sum(1 for summary in summaries if summary.dominant_attribution_type == intended) / len(summaries)
        consistency = 0.0
        if summaries:
            consistency = dominant_counts[dominant_observed] / len(summaries) if dominant_observed else 0.0
        query_summaries.append(
            QueryEvaluationSummary(
                query_id=query_id,
                intended_failure_mode=intended,
                trigger_success_rate=hit_rate,
                dominant_observed_failure_mode=dominant_observed,
                response_count=len(summaries),
                model_consistency=consistency,
                should_keep_as_trigger=hit_rate >= 0.5,
                metadata={},
            )
        )

    return sentence_annotations, response_summaries, query_summaries


def as_dicts(rows: Iterable[object]) -> List[Dict[str, object]]:
    return [asdict(row) for row in rows]
