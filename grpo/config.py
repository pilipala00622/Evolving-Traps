from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class GRPOConfig:
    """
    ms-swift-native GRPO config centered on Qwen2.5-7B-Instruct.

    The generated bundle is meant to be consumed by:
    `swift rlhf --rlhf_type grpo ...`
    """

    model: str = "Qwen/Qwen2.5-7B-Instruct"
    hallusea_dir: str = ""
    output_dir: str = "grpo_output"
    swift_bin: str = "swift"

    train_type: str = "lora"
    torch_dtype: str = "bfloat16"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-6
    warmup_ratio: float = 0.03
    beta: float = 0.03

    num_generations: int = 8
    temperature: float = 0.9
    top_p: float = 0.95
    max_prompt_length: int = 3072
    max_completion_length: int = 512

    dataset_num_proc: int = 2
    dataloader_num_workers: int = 2
    logging_steps: int = 5
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 2
    report_to: str = "none"

    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.5
    deepspeed: str = "zero2"

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    reward_funcs: List[str] = field(default_factory=lambda: ["evohallu_outcome"])
    reward_weights: List[float] = field(default_factory=lambda: [1.0])

    seed: int = 42

    @property
    def max_length(self) -> int:
        return self.max_prompt_length + self.max_completion_length

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["max_length"] = self.max_length
        return payload

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
