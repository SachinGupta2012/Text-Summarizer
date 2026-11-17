"""Launch one resumable Phase 6 QLoRA run under Accelerate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from summarizer_ml.fine_tuning import (
    Phase6Config,
    Phase6RunMode,
    Phase6SelectionManifest,
    gradient_accumulation_steps,
    optimizer_steps,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def finite_losses(history: list[dict[str, Any]]) -> bool:
    for row in history:
        for key, value in row.items():
            if (
                (key == "loss" or key.endswith("_loss"))
                and value is not None
                and not math.isfinite(float(value))
            ):
                return False
    return True


def mode_steps(
    mode: Phase6RunMode,
    manifest: Phase6SelectionManifest,
    config: Phase6Config,
    world_size: int,
) -> int:
    if mode is Phase6RunMode.SMOKE:
        return config.smoke_steps
    if mode is Phase6RunMode.CALIBRATION:
        return config.calibration_steps
    return optimizer_steps(manifest.train_records, config, world_size)


def interval_steps(mode: Phase6RunMode, config: Phase6Config) -> int:
    if mode is Phase6RunMode.SMOKE:
        return max(1, config.smoke_steps // 2)
    if mode is Phase6RunMode.CALIBRATION:
        return config.calibration_steps
    return config.evaluation_steps


def main() -> None:
    arguments = parse_arguments()
    config = Phase6Config.model_validate_json(arguments.config.read_text(encoding="utf-8"))
    selection = Phase6SelectionManifest.model_validate_json(
        arguments.selection_manifest.read_text(encoding="utf-8")
    )
    mode = selection.run_mode
    output_directory = arguments.output_directory
    report_path = output_directory / "training-report.json"
    completed_path = output_directory / "COMPLETED"
    if report_path.is_file() and completed_path.is_file():
        print(f"Phase 6 {mode.value} run is already complete: {report_path}", flush=True)
        return

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    accumulation = gradient_accumulation_steps(config, world_size)
    max_steps = mode_steps(mode, selection, config, world_size)
    interval = interval_steps(mode, config)
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = output_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    train_dataset = load_from_disk(str(arguments.train_cache))
    validation_dataset = load_from_disk(str(arguments.validation_cache))
    if len(train_dataset) != selection.train_records:
        raise RuntimeError("Tokenized training cache count does not match the selection.")
    if len(validation_dataset) != selection.validation_records:
        raise RuntimeError("Tokenized validation cache count does not match the selection.")

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.quantization_type,
        bnb_4bit_use_double_quant=config.use_double_quantization,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=quantization,
        device_map={"": local_rank},
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model_revision = str(getattr(model.config, "_commit_hash", None) or "unknown")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    adapter_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, adapter_config)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    invalid_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if invalid_trainable:
        raise RuntimeError(f"Non-LoRA parameters are trainable: {invalid_trainable[:5]}")

    training_arguments = TrainingArguments(
        output_dir=str(checkpoint_directory),
        overwrite_output_dir=False,
        max_steps=max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=accumulation,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.scheduler,
        max_grad_norm=config.max_grad_norm,
        optim=config.optimizer,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=interval,
        save_strategy="steps",
        save_steps=interval,
        save_total_limit=config.checkpoint_limit,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        dataloader_num_workers=config.dataloader_workers,
        dataloader_prefetch_factor=(
            config.dataloader_prefetch_factor if config.dataloader_workers > 0 else None
        ),
        dataloader_persistent_workers=config.dataloader_workers > 0,
        dataloader_pin_memory=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,
        seed=config.random_seed,
        data_seed=config.random_seed,
        report_to=[],
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding="longest",
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    callbacks = []
    if mode in {Phase6RunMode.PILOT, Phase6RunMode.FULL}:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    last_checkpoint = get_last_checkpoint(str(checkpoint_directory))
    initial_evaluation_metrics = trainer.evaluate(metric_key_prefix="initial")
    torch.cuda.reset_peak_memory_stats(local_rank)
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    evaluation_metrics = trainer.evaluate()
    if not finite_losses(trainer.state.log_history):
        raise RuntimeError("Phase 6 produced a non-finite training or validation loss.")

    if trainer.is_world_process_zero():
        adapter_directory = output_directory / "adapter"
        tokenizer_directory = output_directory / "tokenizer"
        trainer.save_model(str(adapter_directory))
        trainer.save_state()
        tokenizer.save_pretrained(tokenizer_directory)
        history_path = output_directory / "training-history.jsonl"
        with history_path.open("w", encoding="utf-8", newline="\n") as history_file:
            for row in trainer.state.log_history:
                history_file.write(json.dumps(row, sort_keys=True))
                history_file.write("\n")
        runtime = float(train_result.metrics.get("train_runtime", 0.0))
        average_tokens = selection.train_tokens / selection.train_records
        steps_per_second = float(train_result.metrics.get("train_steps_per_second", 0.0))
        approximate_tokens_per_second = (
            steps_per_second
            * config.per_device_train_batch_size
            * world_size
            * accumulation
            * average_tokens
        )
        report = {
            "schema_version": "1.0",
            "run_mode": mode.value,
            "model_id": config.model_id,
            "model_revision": model_revision,
            "config_sha256": sha256(arguments.config),
            "selection_manifest_sha256": sha256(arguments.selection_manifest),
            "train_records": selection.train_records,
            "validation_records": selection.validation_records,
            "world_size": world_size,
            "max_steps": max_steps,
            "completed_steps": trainer.state.global_step,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": (
                config.per_device_train_batch_size * world_size * accumulation
            ),
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "trainable_fraction": trainable_parameters / total_parameters,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(local_rank) / (1024**3),
            "approximate_selected_tokens": selection.train_tokens,
            "approximate_tokens_per_second": (
                approximate_tokens_per_second if runtime > 0 else None
            ),
            "initial_evaluation_metrics": initial_evaluation_metrics,
            "train_metrics": train_result.metrics,
            "evaluation_metrics": evaluation_metrics,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "resumed_from_checkpoint": last_checkpoint,
            "losses_finite": True,
            "adapter_directory": str(adapter_directory),
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(world_size)],
            "library_versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "accelerate",
                    "bitsandbytes",
                    "datasets",
                    "peft",
                    "torch",
                    "transformers",
                )
            },
        }
        atomic_json(report_path, report)
        completed_path.write_text("complete\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
