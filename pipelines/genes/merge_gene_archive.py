import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a previous archive with new population rows above a fitness threshold."
    )
    parser.add_argument("--base-archive", required=True)
    parser.add_argument("--population", required=True)
    parser.add_argument("--output-archive", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--min-fitness", type=float, default=0.75)
    parser.add_argument("--round-label", default="roundX")
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    base_archive = load_jsonl(Path(args.base_archive))
    population = load_jsonl(Path(args.population))

    accepted_children = [
        row for row in population if float(row.get("fitness", 0.0) or 0.0) >= args.min_fitness
    ]

    merged_by_gene: dict[str, dict[str, Any]] = {}
    for row in base_archive + accepted_children:
        gene_id = row.get("gene_id")
        if not gene_id:
            continue
        prev = merged_by_gene.get(gene_id)
        if prev is None or float(row.get("fitness", 0.0) or 0.0) >= float(prev.get("fitness", 0.0) or 0.0):
            merged_by_gene[gene_id] = row

    merged = sorted(
        merged_by_gene.values(),
        key=lambda row: (float(row.get("fitness", 0.0) or 0.0), row.get("gene_id", "")),
        reverse=True,
    )

    summary = {
        "archive_count": len(merged),
        f"{args.round_label}_child_count": len(accepted_children),
        "min_fitness": args.min_fitness,
        "top_gene_ids": [row.get("gene_id") for row in merged[: args.top_k]],
        "top_fitness": [round(float(row.get("fitness", 0.0) or 0.0), 4) for row in merged[: args.top_k]],
    }

    dump_jsonl(Path(args.output_archive), merged)
    dump_json(Path(args.output_summary), summary)


if __name__ == "__main__":
    main()
