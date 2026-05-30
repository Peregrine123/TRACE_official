from data.base_builder import BaseBuilder
from data.base_env import (
    BaseEnv,
    StaticEnv,
    DynamicEnv,
)
from data.agent_safety.builder import SafetyJudgeBuilder

_DATA_BUILDER_MAP = {
    "agent_safety": SafetyJudgeBuilder,
}


def get_data_builder(dataset_cfg) -> BaseBuilder:
    if dataset_cfg.get("name") not in _DATA_BUILDER_MAP:
        raise ValueError("Unsupported dataset.")

    builder_cls = _DATA_BUILDER_MAP[dataset_cfg.get("name")]
    builder = builder_cls(dataset_cfg)

    return builder
