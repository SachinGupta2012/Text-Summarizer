"""Small deterministic calculations used before expensive GPU training."""

import math

from summarizer_ml.fine_tuning.models import Phase6Config, RuntimeProjection


def gradient_accumulation_steps(config: Phase6Config, world_size: int) -> int:
    """Keep the effective batch stable across one or two GPUs."""

    if world_size < 1:
        raise ValueError("World size must be positive.")
    denominator = config.per_device_train_batch_size * world_size
    if config.target_effective_batch_size % denominator:
        raise ValueError("Target effective batch size is not divisible by the GPU micro-batch.")
    return config.target_effective_batch_size // denominator


def optimizer_steps(records: int, config: Phase6Config, world_size: int) -> int:
    """Return optimizer steps required for one complete pass."""

    if records < 1:
        raise ValueError("Training record count must be positive.")
    accumulation = gradient_accumulation_steps(config, world_size)
    effective_batch = config.per_device_train_batch_size * world_size * accumulation
    return math.ceil(records / effective_batch)


def project_runtime(
    measured_steps: int,
    measured_seconds: float,
    projected_steps: int,
) -> RuntimeProjection:
    """Project a bounded run from measured calibration throughput."""

    if measured_steps < 1 or measured_seconds <= 0 or projected_steps < 1:
        raise ValueError("Runtime projection inputs must be positive.")
    seconds_per_step = measured_seconds / measured_steps
    projected_seconds = seconds_per_step * projected_steps
    return RuntimeProjection(
        measured_steps=measured_steps,
        measured_seconds=measured_seconds,
        seconds_per_step=seconds_per_step,
        projected_steps=projected_steps,
        projected_seconds=projected_seconds,
        projected_hours=projected_seconds / 3_600,
    )

