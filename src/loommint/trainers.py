from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class SFTTrainingConfig:
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 2048
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    logging_steps: int = 10
    seed: int = 42
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    load_in_4bit: bool = False

    def __post_init__(self) -> None:
        for name in ("learning_rate", "num_train_epochs", "lora_dropout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
        if self.learning_rate <= 0 or self.num_train_epochs <= 0:
            raise ValueError("learning_rate and num_train_epochs must be positive")
        for name in ("per_device_train_batch_size", "gradient_accumulation_steps", "max_length", "lora_rank", "lora_alpha", "logging_steps", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if name != "seed" and value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be between 0 and 1")
        if not isinstance(self.load_in_4bit, bool):
            raise ValueError("load_in_4bit must be a boolean")
        if not self.target_modules or any(not isinstance(module, str) or not module for module in self.target_modules):
            raise ValueError("target_modules must contain non-empty strings")

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any]) -> SFTTrainingConfig:
        if not isinstance(parameters, dict):
            raise ValueError("SFT parameters must be an object")
        aliases = {"r": "lora_rank", "alpha": "lora_alpha"}
        normalized = {}
        for key, value in parameters.items():
            if not isinstance(key, str):
                raise ValueError("SFT parameter names must be strings")
            normalized_key = aliases.get(key, key)
            if normalized_key in normalized:
                raise ValueError(f"SFT parameter '{normalized_key}' was provided more than once")
            normalized[normalized_key] = value
        allowed = cls.__dataclass_fields__.keys()
        unknown = normalized.keys() - allowed
        if unknown:
            raise ValueError(f"unsupported SFT parameters: {', '.join(sorted(unknown))}")
        if "target_modules" in normalized:
            modules = normalized["target_modules"]
            if isinstance(modules, str):
                normalized["target_modules"] = (modules,)
            elif isinstance(modules, (list, tuple)):
                normalized["target_modules"] = tuple(modules)
            else:
                raise ValueError("target_modules must be a string or list of strings")
        return cls(**normalized)


@dataclass(frozen=True)
class TrainingRequest:
    model_id: str
    train_file: Path
    validation_file: Path
    output_dir: Path
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingResult:
    backend: str
    output_dir: Path
    metrics: dict[str, float]


@runtime_checkable
class TrainerBackend(Protocol):
    name: str
    capabilities: frozenset[str]

    def train(self, request: TrainingRequest) -> TrainingResult:
        ...
