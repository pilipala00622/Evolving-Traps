"""Compatibility shim. Use `python3 -m human_review.cli ...` instead."""

from human_review.cli import main


if __name__ == "__main__":
    main()
