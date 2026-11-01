"""Tests for the dependency-light Phase 1 ML package."""

from summarizer_ml import MlProjectConfig


def test_project_configuration_documents_agreed_strategy() -> None:
    config = MlProjectConfig()

    assert config.base_model_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.fine_tuning_method == "qlora"
    assert config.training_platform == "kaggle"
    assert config.accelerator == "gpu-t4"
