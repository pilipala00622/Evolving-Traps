from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "data/知识库评测共建-for混元-450条 copy.jsonl"
DEFAULT_OUTPUT = "data/hard_hallucination/source_contexts.jsonl"


def _normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
                continue
            if isinstance(item, dict):
                value = item.get("value") or item.get("text") or item.get("content") or item.get("input") or ""
                if value:
                    parts.append(str(value).strip())
        return "\n".join(part for part in parts if part)
    return str(content).strip()


def _load_request_body(raw_body: Any) -> dict[str, Any]:
    if isinstance(raw_body, dict):
        return raw_body
    if isinstance(raw_body, str):
        return json.loads(raw_body)
    raise TypeError(f"Unsupported requests_body type: {type(raw_body).__name__}")


def _split_prompt_wrapper(text: str) -> tuple[str, str]:
    doc_text = text.strip()
    question_text = ""
    if "[文档内容]：" in doc_text:
        _, doc_text = doc_text.split("[文档内容]：", 1)
        doc_text = doc_text.strip()
    for marker in ["\n[问题]:", "\n[问题]：", "[问题]:", "[问题]：", "\n问题:", "\n问题：", "问题:", "问题："]:
        if marker in doc_text:
            doc_text, question_text = doc_text.split(marker, 1)
            return doc_text.strip(), question_text.strip()
    return doc_text, question_text


def extract_records(input_path: Path, first_user_only: bool = True, document_only: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            body = _load_request_body(row.get("requests_body"))
            user_texts: list[str] = []
            for message in body.get("messages", []):
                if message.get("role") != "user":
                    continue
                text = _normalize_content(message.get("content"))
                if text:
                    user_texts.append(text)
                    if first_user_only:
                        break
            if not user_texts:
                continue
            primary = user_texts[0]
            context_text, source_question_text = _split_prompt_wrapper(primary) if document_only else (primary, "")
            records.append(
                {
                    "trace_id": row.get("trace_id"),
                    "knowledge_base_category": row.get("知识库分类"),
                    "query_category": row.get("query类别"),
                    "source_query": row.get("query"),
                    "source_question_text": source_question_text,
                    "context_text": context_text,
                    "context_length": len(context_text),
                    "source_model": body.get("model"),
                    "source_query_id": body.get("query_id"),
                }
            )
    return records


def write_jsonl(output_path: Path, records: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract clean document-only contexts for hard hallucination generation")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--first-user-only", action="store_true", default=True)
    parser.add_argument("--document-only", action="store_true", default=True)
    args = parser.parse_args()

    records = extract_records(
        Path(args.input),
        first_user_only=args.first_user_only,
        document_only=args.document_only,
    )
    write_jsonl(Path(args.output), records)
    summary = {
        "input_path": args.input,
        "output_path": args.output,
        "record_count": len(records),
        "avg_context_length": round(sum(r["context_length"] for r in records) / max(1, len(records)), 2),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
