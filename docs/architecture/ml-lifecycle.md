# ML lifecycle

## Artifact progression

```text
source manifest
  -> processed dataset version
  -> fixed evaluation version
  -> base-model benchmark
  -> QLoRA experiment
  -> adapter benchmark
  -> promotion decision
  -> inference artifact
```

## Reproducibility requirements

Every dataset build records source identifiers, licenses, preprocessing version,
quality flags, split assignment, and token statistics. Every experiment records
the base model revision, dataset version, prompt template, random seed, QLoRA
configuration, environment, metrics, and artifact location.

## Fair comparison

The base model and adapter use the same held-out examples, prompt, decoding
configuration, and metric implementation. Results are reported overall and by
domain. Automatic metrics are supplemented by entity preservation, factuality,
human preference, latency, throughput, and memory measurements.

## Kaggle boundary

Kaggle provides temporary T4 GPU compute. Repository code remains canonical.
Notebooks install the pinned package, validate the environment, invoke smoke or
training commands, save resumable checkpoints, and export only approved artifacts.
