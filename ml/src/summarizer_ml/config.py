"""Phase-independent identifiers for later ML experiments."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class MlProjectConfig(BaseModel):
    """Describe the agreed model strategy without loading model dependencies."""

    model_config = ConfigDict(frozen=True)

    base_model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    fine_tuning_method: Literal["qlora"] = "qlora"
    training_platform: Literal["kaggle"] = "kaggle"
    accelerator: Literal["gpu-t4"] = "gpu-t4"
