"""CLI entrypoint for the human review subsystem."""

from __future__ import annotations

import argparse
from pathlib import Path

from human_review.review_logic import export_review_tasks, merge_reviews


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="人工检查和标注 benchmark candidate 的辅助脚本")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="导出待审 JSONL 任务")
    export_parser.add_argument("--input", required=True, help="benchmark candidate JSON 文件")
    export_parser.add_argument("--output", required=True, help="导出的 review task JSONL 文件")
    export_parser.add_argument(
        "--only-pending",
        action="store_true",
        help="只导出尚未 approved 的样本",
    )

    merge_parser = subparsers.add_parser("merge", help="把人工标注结果回写到 benchmark candidate")
    merge_parser.add_argument("--input", required=True, help="原始 benchmark candidate JSON 文件")
    merge_parser.add_argument("--reviews", required=True, help="人工填写后的 review task JSONL")
    merge_parser.add_argument("--output", required=True, help="回写后的 benchmark candidate JSON 文件")
    merge_parser.add_argument(
        "--approved-output",
        default="",
        help="可选。单独导出 approve 的样本子集",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "export":
        export_review_tasks(
            input_path=Path(args.input),
            output_path=Path(args.output),
            only_pending=args.only_pending,
        )
        return

    if args.command == "merge":
        approved_output = Path(args.approved_output) if args.approved_output else None
        merge_reviews(
            input_path=Path(args.input),
            reviews_path=Path(args.reviews),
            output_path=Path(args.output),
            approved_output_path=approved_output,
        )
        return

    raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
