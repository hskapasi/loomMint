from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

from .trainers import SFTTrainingConfig, TrainingRequest, TrainingResult


class TrainerDependencyError(RuntimeError):
    pass


class TrainerConfigurationError(ValueError):
    pass


class TRLSFTBackend:
    name = "trl-sft"
    capabilities = frozenset({"sft", "lora"})
    _use_unsloth = False

    def train(self, request: TrainingRequest) -> TrainingResult:
        config = SFTTrainingConfig.from_parameters(request.parameters)
        self._validate_request(request, config)
        runtime = self._load_runtime()
        train_dataset = runtime.datasets.load_dataset("json", data_files=str(request.train_file), split="train")
        eval_dataset = runtime.datasets.load_dataset("json", data_files=str(request.validation_file), split="train")
        self._validate_dataset(train_dataset, "training")
        self._validate_dataset(eval_dataset, "validation")
        model, tokenizer, peft_config = self._load_model(request, config, runtime)
        if not getattr(tokenizer, "chat_template", None):
            raise TrainerConfigurationError("the selected model tokenizer has no chat template for conversational data")
        training_args = runtime.trl.SFTConfig(
            output_dir=str(request.output_dir),
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_length=config.max_length,
            logging_steps=config.logging_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            report_to="none",
            seed=config.seed,
        )
        trainer = runtime.trl.SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        request.output_dir.mkdir(parents=True, exist_ok=True)
        train_result = trainer.train()
        metrics = self._numeric_metrics(getattr(train_result, "metrics", {}))
        metrics.update(self._numeric_metrics(trainer.evaluate()))
        trainer.save_model(str(request.output_dir))
        tokenizer.save_pretrained(str(request.output_dir))
        return TrainingResult(self.name, request.output_dir, metrics)

    def _validate_request(self, request: TrainingRequest, config: SFTTrainingConfig) -> None:
        if not request.model_id.strip():
            raise TrainerConfigurationError("model_id must not be empty")
        if not request.train_file.is_file() or not request.validation_file.is_file():
            raise TrainerConfigurationError("training and validation files must exist")
        if request.output_dir.exists() and (not request.output_dir.is_dir() or any(request.output_dir.iterdir())):
            raise TrainerConfigurationError("output_dir must be new or an empty directory")
        if config.load_in_4bit and not self._use_unsloth:
            raise TrainerConfigurationError("4-bit loading requires the Unsloth SFT backend")

    def _load_runtime(self) -> Any:
        try:
            datasets = importlib.import_module("datasets")
            peft = importlib.import_module("peft")
            transformers = importlib.import_module("transformers")
            trl = importlib.import_module("trl")
        except ImportError as exc:
            extra = "unsloth" if self._use_unsloth else "trl"
            raise TrainerDependencyError(f'install training dependencies from the repository root with `python -m pip install -e ".[{extra}]"`') from exc
        runtime = SimpleNamespace(datasets=datasets, peft=peft, transformers=transformers, trl=trl)
        if self._use_unsloth:
            try:
                runtime.unsloth = importlib.import_module("unsloth")
            except ImportError as exc:
                raise TrainerDependencyError('install acceleration dependencies from the repository root with `python -m pip install -e ".[unsloth]"`') from exc
        return runtime

    def _load_model(self, request: TrainingRequest, config: SFTTrainingConfig, runtime: Any) -> tuple[Any, Any, Any]:
        tokenizer = runtime.transformers.AutoTokenizer.from_pretrained(request.model_id)
        if not getattr(tokenizer, "chat_template", None):
            raise TrainerConfigurationError("the selected model tokenizer has no chat template for conversational data")
        model = runtime.transformers.AutoModelForCausalLM.from_pretrained(request.model_id, dtype="auto")
        peft_config = runtime.peft.LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(config.target_modules),
        )
        return model, tokenizer, peft_config

    @staticmethod
    def _validate_dataset(dataset: Any, label: str) -> None:
        if not len(dataset):
            raise TrainerConfigurationError(f"{label} dataset is empty")
        if "messages" not in getattr(dataset, "column_names", []):
            raise TrainerConfigurationError(f"{label} dataset must contain canonical 'messages' records")

    @staticmethod
    def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
        return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}


class UnslothSFTBackend(TRLSFTBackend):
    name = "trl-sft-unsloth"
    capabilities = frozenset({"sft", "lora", "qlora"})
    _use_unsloth = True

    def _load_model(self, request: TrainingRequest, config: SFTTrainingConfig, runtime: Any) -> tuple[Any, Any, None]:
        model, tokenizer = runtime.unsloth.FastLanguageModel.from_pretrained(
            model_name=request.model_id,
            max_seq_length=config.max_length,
            dtype="auto",
            load_in_4bit=config.load_in_4bit,
        )
        model = runtime.unsloth.FastLanguageModel.get_peft_model(
            model,
            r=config.lora_rank,
            target_modules=list(config.target_modules),
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=config.seed,
        )
        return model, tokenizer, None

