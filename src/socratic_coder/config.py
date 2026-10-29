"""Typed configuration management for SocraticCoder-7B."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class HardwareConfig:
    device: str = "cuda"
    tutor_device: str = "cuda:0"
    student_device: str = "auto"
    precision: str = "bf16"
    use_4bit_quantization: bool = True
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    num_verify_workers: int = 8


@dataclass
class ModelsConfig:
    tutor_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    student_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    eval_student_models: List[str] = field(
        default_factory=lambda: [
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-Coder-3B-Instruct",
        ]
    )


@dataclass
class LoRAConfig:
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class DatasetConfig:
    num_train_pairs: int = 500
    num_test_pairs: int = 120
    num_sft_trajectories: int = 500
    max_dialogue_turns: int = 3
    adversarial_ratio: float = 0.30
    filter_zero_variance: bool = True
    sources: List[str] = field(
        default_factory=lambda: [
            "humanevalfix",
            "mbppplus",
            "curated_algorithmic_bank",
        ]
    )


@dataclass
class SFTConfig:
    num_epochs: int = 2
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.0e-4
    warmup_ratio: float = 0.08
    weight_decay: float = 0.01
    max_seq_length: int = 1536
    save_steps: int = 50


@dataclass
class RLConfig:
    algorithm: str = "grpo_dapo"
    num_steps: int = 250
    group_size: int = 6
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1.5e-5
    kl_coef: float = 0.04
    clip_eps_low: float = 0.20
    clip_eps_high: float = 0.28
    dapo_dynamic_sampling: bool = True
    max_resample_attempts: int = 3
    max_tutor_new_tokens: int = 220
    max_student_new_tokens: int = 384
    temperature: float = 0.85
    top_p: float = 0.92
    save_every_steps: int = 50


@dataclass
class RewardConfig:
    alpha: float = 0.40
    beta: float = 0.60
    gamma: float = 0.0005
    gate_first_short_circuit: bool = True


@dataclass
class LeakageGateConfig:
    max_code_lines: int = 2
    require_diagnostic_question: bool = True
    max_ngram_overlap: float = 0.35
    ngram_size: int = 4


@dataclass
class EvaluationConfig:
    max_test_turns: int = 3
    best_of_n: int = 6
    test_timeout_sec: float = 2.0


@dataclass
class SocraticCoderConfig:
    project_name: str = "SocraticCoder-7B"
    seed: int = 42
    output_dir: str = "outputs/kaggle_t4x2"
    data_dir: str = "data/processed"
    results_dir: str = "results"
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    leakage_gate: LeakageGateConfig = field(default_factory=LeakageGateConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SocraticCoderConfig":
        def _filter_dataclass_kwargs(dc_type: Any, data: Dict[str, Any]) -> Dict[str, Any]:
            valid_keys = {f.name for f in dc_type.__dataclass_fields__.values()}
            return {k: v for k, v in (data or {}).items() if k in valid_keys}

        return cls(
            project_name=raw.get("project_name", "SocraticCoder-7B"),
            seed=int(raw.get("seed", 42)),
            output_dir=raw.get("output_dir", "outputs/kaggle_t4x2"),
            data_dir=raw.get("data_dir", "data/processed"),
            results_dir=raw.get("results_dir", "results"),
            hardware=HardwareConfig(**_filter_dataclass_kwargs(HardwareConfig, raw.get("hardware", {}))),
            models=ModelsConfig(**_filter_dataclass_kwargs(ModelsConfig, raw.get("models", {}))),
            lora=LoRAConfig(**_filter_dataclass_kwargs(LoRAConfig, raw.get("lora", {}))),
            dataset=DatasetConfig(**_filter_dataclass_kwargs(DatasetConfig, raw.get("dataset", {}))),
            sft=SFTConfig(**_filter_dataclass_kwargs(SFTConfig, raw.get("sft", {}))),
            rl=RLConfig(**_filter_dataclass_kwargs(RLConfig, raw.get("rl", {}))),
            reward=RewardConfig(**_filter_dataclass_kwargs(RewardConfig, raw.get("reward", {}))),
            leakage_gate=LeakageGateConfig(**_filter_dataclass_kwargs(LeakageGateConfig, raw.get("leakage_gate", {}))),
            evaluation=EvaluationConfig(**_filter_dataclass_kwargs(EvaluationConfig, raw.get("evaluation", {}))),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SocraticCoderConfig":
        text = Path(path).read_text(encoding="utf-8")
        try:
            import yaml  # type: ignore

            raw = yaml.safe_load(text) or {}
        except ImportError:
            raw = _minimal_yaml_parse(text)
        return cls.from_dict(raw)


def _minimal_yaml_parse(text: str) -> Dict[str, Any]:
    """Fallback lightweight parser for 2-level YAML configs when PyYAML is absent."""
    result: Dict[str, Any] = {}
    current_section: Optional[str] = None
    current_list_key: Optional[str] = None

    def _cast_scalar(val: str) -> Any:
        val = val.strip()
        if val.startswith(('"', "'")) and val.endswith(('"', "'")):
            return val[1:-1]
        if val.lower() == "true":
            return True
        if val.lower() == "false":
            return False
        try:
            if "." in val or "e-" in val.lower() or "e+" in val.lower():
                return float(val)
            return int(val)
        except ValueError:
            return val

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if indent == 0 and ":" in stripped:
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip()
            current_list_key = None
            if not val:
                current_section = key
                result[current_section] = {}
            else:
                current_section = None
                result[key] = _cast_scalar(val)
        elif indent > 0 and current_section is not None:
            if stripped.startswith("- "):
                item_val = _cast_scalar(stripped[2:])
                if current_list_key is not None:
                    result[current_section].setdefault(current_list_key, []).append(item_val)
            elif ":" in stripped:
                key, val = stripped.split(":", 1)
                key = key.strip()
                val = val.strip()
                if not val:
                    current_list_key = key
                    result[current_section][current_list_key] = []
                else:
                    current_list_key = None
                    result[current_section][key] = _cast_scalar(val)

    return result
