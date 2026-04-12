#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List

try:
    from grpo.config import GRPOConfig
    from grpo.dataset_builder import build_ms_swift_dataset_bundle
except ModuleNotFoundError:
    from config import GRPOConfig
    from dataset_builder import build_ms_swift_dataset_bundle


def _bool_flag(value: bool) -> str:
    return "true" if value else "false"


def detect_ms_swift_launcher(swift_bin: str) -> bool:
    resolved = shutil.which(swift_bin)
    if not resolved:
        return False
    try:
        proc = subprocess.run(
            [resolved, "rlhf", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return proc.returncode == 0


def build_ms_swift_command(config: GRPOConfig, dataset_path: Path, plugin_path: Path) -> List[str]:
    command = [
        config.swift_bin,
        "rlhf",
        "--rlhf_type",
        "grpo",
        "--model",
        config.model,
        "--dataset",
        str(dataset_path),
        "--external_plugins",
        str(plugin_path),
        "--train_type",
        config.train_type,
        "--torch_dtype",
        config.torch_dtype,
        "--output_dir",
        config.output_dir,
        "--num_train_epochs",
        str(config.num_train_epochs),
        "--per_device_train_batch_size",
        str(config.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(config.gradient_accumulation_steps),
        "--learning_rate",
        str(config.learning_rate),
        "--warmup_ratio",
        str(config.warmup_ratio),
        "--beta",
        str(config.beta),
        "--num_generations",
        str(config.num_generations),
        "--temperature",
        str(config.temperature),
        "--top_p",
        str(config.top_p),
        "--max_length",
        str(config.max_length),
        "--max_completion_length",
        str(config.max_completion_length),
        "--dataset_num_proc",
        str(config.dataset_num_proc),
        "--dataloader_num_workers",
        str(config.dataloader_num_workers),
        "--logging_steps",
        str(config.logging_steps),
        "--save_steps",
        str(config.save_steps),
        "--eval_steps",
        str(config.eval_steps),
        "--save_total_limit",
        str(config.save_total_limit),
        "--report_to",
        config.report_to,
        "--deepspeed",
        config.deepspeed,
        "--use_vllm",
        _bool_flag(config.use_vllm),
        "--vllm_mode",
        config.vllm_mode,
        "--vllm_gpu_memory_utilization",
        str(config.vllm_gpu_memory_utilization),
        "--lora_rank",
        str(config.lora_rank),
        "--lora_alpha",
        str(config.lora_alpha),
        "--lora_dropout",
        str(config.lora_dropout),
        "--seed",
        str(config.seed),
    ]
    command.extend(["--reward_funcs", *config.reward_funcs])
    command.extend(["--reward_weights", *(str(weight) for weight in config.reward_weights)])
    return command


def command_to_shell(command: List[str]) -> str:
    return " \\\n  ".join(shlex.quote(part) for part in command)


def write_launch_bundle(config: GRPOConfig, output_dir: Path, dataset_path: Path) -> Path:
    plugin_path = (Path(__file__).resolve().parent / "ms_swift_plugin.py").resolve()
    command = build_ms_swift_command(config, dataset_path, plugin_path)

    run_script_path = output_dir / "run_ms_swift_grpo.sh"
    command_tail = command[1:]
    run_script_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        f"DEFAULT_SWIFT_RLHF_BIN={shlex.quote(config.swift_bin)}\n"
        "SWIFT_RLHF_BIN=\"${SWIFT_RLHF_BIN:-$DEFAULT_SWIFT_RLHF_BIN}\"\n"
        f"cd {shlex.quote(str(Path(__file__).resolve().parents[1]))}\n\n"
        f"\"$SWIFT_RLHF_BIN\" \\\n  {command_to_shell(command_tail)}\n",
        encoding="utf-8",
    )
    run_script_path.chmod(0o755)

    launch_note = {
        "framework": "ms-swift",
        "rlhf_type": "grpo",
        "model": config.model,
        "dataset_path": str(dataset_path),
        "external_plugin": str(plugin_path),
        "reward_funcs": config.reward_funcs,
        "reward_weights": config.reward_weights,
        "run_script": str(run_script_path),
        "launch_command": command,
        "swift_bin": config.swift_bin,
        "ms_swift_launcher_detected": detect_ms_swift_launcher(config.swift_bin),
    }
    path = output_dir / "ms_swift_launch_manifest.json"
    path.write_text(json.dumps(launch_note, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an ms-swift GRPO training bundle from HalluSEA outputs.")
    parser.add_argument("--hallusea-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--swift-bin", default="swift")
    parser.add_argument("--group-size", type=int, default=8, dest="num_generations")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--dry-run", action="store_true", help="Only prepare dataset/config files.")
    parser.add_argument("--run", action="store_true", help="Run `swift rlhf` after preparing the bundle.")
    args = parser.parse_args()

    config = GRPOConfig(
        model=args.base_model,
        hallusea_dir=args.hallusea_dir,
        output_dir=args.output_dir,
        swift_bin=args.swift_bin,
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        beta=args.beta,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "ms_swift_grpo_config.json"
    config.write(config_path)
    dataset_summary = build_ms_swift_dataset_bundle(Path(config.hallusea_dir), output_dir)
    dataset_path = output_dir / "ms_swift_grpo_dataset.jsonl"
    manifest_path = write_launch_bundle(config, output_dir, dataset_path)

    status = "prepared"
    if args.run:
        if not detect_ms_swift_launcher(config.swift_bin):
            raise SystemExit(
                f"`{config.swift_bin}` is not a usable ms-swift launcher. "
                "Please point --swift-bin to the ms-swift CLI before using --run."
            )
        command = build_ms_swift_command(
            config,
            dataset_path=dataset_path,
            plugin_path=(Path(__file__).resolve().parent / "ms_swift_plugin.py").resolve(),
        )
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])
        status = "executed"
    elif not args.dry_run:
        status = "prepared_only"

    payload = {
        "status": status,
        "framework": "ms-swift",
        "config_path": str(config_path),
        "dataset_summary": dataset_summary,
        "launch_manifest": str(manifest_path),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
