from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import re
import threading
from pathlib import Path
from typing import Any

from llm import get_llm_instance
from tqdm import tqdm

from hard_hallucination_config import (
    DEFAULT_HARD_HALLUCINATION_FAMILY_CODES,
    HARD_HALLUCINATION_FAMILIES,
)


DEFAULT_INPUT = "data/hard_hallucination/source_contexts.jsonl"
DEFAULT_OUTPUT = "data/hard_hallucination/hard_hallucination_cards.jsonl"
DEFAULT_MODEL = "gpt-5.4"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _extract_json_candidate(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    candidates = [text]
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates.extend(fenced)
    brace_match = re.search(r"\{.*\}", text, flags=re.S)
    if brace_match:
        candidates.append(brace_match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _load_guides() -> str:
    guide_paths = [
        Path("reference/hard_hallucination_主线方案_v1.md"),
        Path("reference/hard_hallucination_出题规范_v1.md"),
        Path("reference/hard_hallucination_标注协议_v1.md"),
    ]
    return "\n\n".join(path.read_text(encoding="utf-8") for path in guide_paths)


def _allowed_family_text() -> str:
    lines = []
    for code in DEFAULT_HARD_HALLUCINATION_FAMILY_CODES:
        meta = HARD_HALLUCINATION_FAMILIES[code]
        lines.append(
            f"- {meta['label']} ({code})：{meta['definition']}；安全行为：{meta['expected_safe_behavior']}；主错误归因：{meta['target_failure_mode']}"
        )
    return "\n".join(lines)


def _build_prompt(record: dict[str, Any], guides_text: str, directions_per_context: int) -> str:
    schema = {
        "decision": "keep",
        "rejection_reason": "",
        "question_cards": [
            {
                "query": "string",
                "hard_hallucination_family": "weak_evidence_overclaim|missing_key_info_hard_answer|background_as_direct_evidence",
                "target_failure_mode": "越权推理|无中生有|生成错误",
                "target_failure_subtype": "string",
                "trigger_mechanism_label": "string",
                "trigger_mechanism_code": "string",
                "expected_safe_behavior": "string",
                "why_this_is_hallucination": "string",
                "evidence_source_hint": ["string"],
                "judge_anchor": "string",
                "author_self_check": {
                    "is_real_hallucination_trigger": True,
                    "boundary_is_clear": True,
                    "not_simple_extraction": True,
                    "why_it_should_induce_hallucination": "string"
                }
            }
        ]
    }

    return f"""
你是一个专门设计 hard hallucination benchmark 的研究助手。

任务目标：
不是生成普通检索题，也不是生成单纯的抽取难题，而是要生成真正容易暴露 hallucination 的问题。

你只能围绕下面三类 family 出题：
{_allowed_family_text()}

必须遵守：
1. query 必须清楚自然，像真实用户问题。
2. 难度不能来自表述混乱，必须来自证据边界危险。
3. 问题必须让模型正确行为是“克制、保留、拒绝越界、只用直接支持证据”。
4. 不要生成纯集合枚举、纯格式抽取、纯数值计算题，除非它们真的体现 hallucination 风险。
5. 不要生成开放总结、主观建议、感悟、泛解释题。
6. question_cards 数量固定为 {directions_per_context}。
7. 如果这个 context 根本不适合 hard hallucination 出题，输出 decision=reject。

补充协议摘要：
{guides_text[:6000]}

[Context Metadata]
- trace_id: {record.get('trace_id')}
- knowledge_base_category: {record.get('knowledge_base_category')}
- query_category: {record.get('query_category')}
- source_question: {record.get('source_query')}

[Context]
{record.get('context_text')}

请严格输出 JSON，不要附加解释。格式如下：
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


def _validate_card(card: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    family = str(card.get("hard_hallucination_family", "")).strip()
    if family not in HARD_HALLUCINATION_FAMILIES:
        errors.append("invalid hard_hallucination_family")
    if not str(card.get("query", "")).strip():
        errors.append("empty query")
    if not str(card.get("target_failure_mode", "")).strip():
        errors.append("missing target_failure_mode")
    if not str(card.get("expected_safe_behavior", "")).strip():
        errors.append("missing expected_safe_behavior")
    if not str(card.get("judge_anchor", "")).strip():
        errors.append("missing judge_anchor")
    if not str(card.get("why_this_is_hallucination", "")).strip():
        errors.append("missing why_this_is_hallucination")
    self_check = card.get("author_self_check") or {}
    if self_check.get("not_simple_extraction") is not True:
        errors.append("not_simple_extraction is not true")
    return errors


def generate_cards(
    input_path: Path | str,
    output_path: Path | str,
    model_name: str = DEFAULT_MODEL,
    directions_per_context: int = 2,
    limit: int = 0,
    max_workers: int = 5,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = _load_jsonl(input_path)
    if limit > 0:
        records = records[:limit]
    guides_text = _load_guides()
    lock = threading.Lock()

    def worker(record: dict[str, Any]) -> dict[str, Any]:
        llm = get_llm_instance(model_name)
        prompt = _build_prompt(record, guides_text, directions_per_context)
        answer = llm.get_model_answer(prompt=prompt, history=[])
        parsed = _extract_json_candidate(answer)
        result = {
            "trace_id": record.get("trace_id"),
            "knowledge_base_category": record.get("knowledge_base_category"),
            "query_category": record.get("query_category"),
            "source_query": record.get("source_query"),
            "source_question_text": record.get("source_question_text"),
            "context_length": record.get("context_length"),
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "model_name": model_name,
            "raw_answer": answer,
            "parsed_payload": parsed,
            "status": "ok" if parsed else "parse_error",
        }
        if parsed:
            cards = parsed.get("question_cards") or []
            result["decision"] = parsed.get("decision", "")
            result["rejection_reason"] = parsed.get("rejection_reason", "")
            result["question_cards"] = cards
            result["validation_errors"] = [_validate_card(card) for card in cards]
        else:
            result["question_cards"] = []
            result["validation_errors"] = []
        return result

    with output_path.open("w", encoding="utf-8") as sink:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, row) for row in records]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="hard-hallu generation"):
                row = future.result()
                with lock:
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hard hallucination question cards")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--directions-per-context", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=5)
    args = parser.parse_args()

    generate_cards(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        directions_per_context=args.directions_per_context,
        limit=args.limit,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
