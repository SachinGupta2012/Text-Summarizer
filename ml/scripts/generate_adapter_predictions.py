"""Generate resumable adapter predictions on one independent GPU worker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import monotonic, perf_counter

import torch
from peft import PeftModel
from summarizer_ml.evaluation import (
    GeneratedSummary,
    Phase5Report,
    Phase7Config,
    Phase7Stage,
    generate_baseline_predictions,
    load_benchmark_selection,
    select_phase7_examples,
    select_worker_examples,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase5-report", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--adapter-directory", type=Path, required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--phase7-config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(item.value for item in Phase7Stage), required=True)
    parser.add_argument("--record-ids", type=Path)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--inference-batch-size", type=int)
    parser.add_argument("--checkpoint-subdirectory", default="workers")
    return parser.parse_args()


def load_record_ids(path: Path) -> tuple[str, ...]:
    """Load unique record IDs from JSONL or one-ID-per-line input."""

    record_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line) if line.lstrip().startswith("{") else line.strip()
        record_id = value.get("record_id") if isinstance(value, dict) else value
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"Invalid record ID entry in {path}.")
        record_ids.append(record_id)
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Record ID input contains duplicates.")
    return tuple(record_ids)


class QwenAdapterGenerator:
    """One-device Qwen generator with a frozen PEFT adapter."""

    def __init__(self, model: PeftModel, tokenizer: AutoTokenizer, device: torch.device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        revision = getattr(self.model.config, "_commit_hash", None)
        self._model_revision = str(revision or "unknown")

    @property
    def model_revision(self) -> str:
        return self._model_revision

    def generate(self, examples):
        texts = [
            self.tokenizer.apply_chat_template(
                [message.model_dump(mode="json") for message in example.prompt],
                tokenize=False,
                add_generation_prompt=True,
            )
            for example in examples
        ]
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_width = encoded["input_ids"].shape[1]
        torch.cuda.synchronize(self.device)
        started = perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        torch.cuda.synchronize(self.device)
        latency = (perf_counter() - started) / len(examples)
        results = []
        for row in output[:, input_width:]:
            token_ids = row.tolist()
            if self.tokenizer.eos_token_id in token_ids:
                end = token_ids.index(self.tokenizer.eos_token_id) + 1
                token_ids = token_ids[:end]
            results.append(
                GeneratedSummary(
                    text=self.tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
                    generated_tokens=len(token_ids),
                    latency_seconds=latency,
                )
            )
        return tuple(results)

    max_new_tokens: int = 1_024


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 7 adapter generation requires a CUDA GPU.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.device_count() < world_size:
        raise RuntimeError("Visible CUDA devices are fewer than inference workers.")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    phase5_report = Phase5Report.model_validate_json(
        arguments.phase5_report.read_text(encoding="utf-8")
    )
    phase7_config = Phase7Config.model_validate_json(
        arguments.phase7_config.read_text(encoding="utf-8")
    )
    stage = Phase7Stage(arguments.stage)
    full_selection = load_benchmark_selection(arguments.selection)
    stage_selection = select_phase7_examples(full_selection, phase7_config, stage)
    if arguments.record_ids is not None:
        requested = set(load_record_ids(arguments.record_ids))
        available = {example.record_id for example in stage_selection}
        missing = requested.difference(available)
        if missing:
            raise ValueError(f"Requested record IDs are outside the selection: {len(missing)}")
        stage_selection = tuple(
            example for example in stage_selection if example.record_id in requested
        )
    worker_selection = select_worker_examples(stage_selection, local_rank, world_size)
    inference_batch_size = arguments.inference_batch_size or phase7_config.inference_batch_size
    max_new_tokens = arguments.max_new_tokens or phase5_report.config.max_new_tokens
    if inference_batch_size < 1 or inference_batch_size > 16:
        raise ValueError("Inference batch size must be between 1 and 16.")
    if max_new_tokens < 64 or max_new_tokens > 4_096:
        raise ValueError("Maximum new tokens must be between 64 and 4096.")
    generation_config = phase5_report.config.model_copy(
        update={
            "inference_batch_size": inference_batch_size,
            "max_new_tokens": max_new_tokens,
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer_directory)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=generation_config.load_in_4bit,
        bnb_4bit_quant_type=generation_config.quantization_type,
        bnb_4bit_use_double_quant=generation_config.use_double_quantization,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        generation_config.model_id,
        device_map={"": local_rank},
        quantization_config=quantization,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        arguments.adapter_directory,
        is_trainable=False,
    )
    model.eval()
    generator = QwenAdapterGenerator(model, tokenizer, device)
    generator.max_new_tokens = generation_config.max_new_tokens
    worker_directory = arguments.output_directory / arguments.checkpoint_subdirectory
    worker_directory.mkdir(parents=True, exist_ok=True)
    prediction_path = worker_directory / f"rank-{local_rank:02d}.jsonl"
    started = monotonic()

    def report_progress(stage_name: str, count: int, total: int) -> None:
        elapsed_minutes = (monotonic() - started) / 60
        rate = count / max(elapsed_minutes, 1e-6)
        remaining = (total - count) / max(rate, 1e-6)
        print(
            f"Phase 7 rank={local_rank} {stage_name}: {count:,}/{total:,}; "
            f"elapsed={elapsed_minutes:.1f}m; eta={remaining:.1f}m",
            flush=True,
        )

    predictions = generate_baseline_predictions(
        worker_selection,
        prediction_path,
        generator,
        generation_config,
        progress_every=10,
        progress_callback=report_progress,
        allow_checkpoint_superset=True,
    )
    metadata_path = worker_directory / f"rank-{local_rank:02d}-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "stage": stage.value,
                "rank": local_rank,
                "world_size": world_size,
                "records": len(predictions),
                "max_new_tokens": generation_config.max_new_tokens,
                "inference_batch_size": generation_config.inference_batch_size,
                "model_revision": generator.model_revision,
                "gpu": torch.cuda.get_device_name(local_rank),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Phase 7 rank {local_rank} complete: {len(predictions):,} records", flush=True)


if __name__ == "__main__":
    main()
