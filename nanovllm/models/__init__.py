"""Model package exports with lazy imports for non-GPU tooling."""

import importlib

__all__ = [
    "Qwen3ForCausalLM",
    "Cpm4ForCausalLM",
    "model_registry",
    "register_model",
    "get_model_class",
    "create_model",
    "list_supported_models",
]


def __getattr__(name):
    if name == "Qwen3ForCausalLM":
        from nanovllm.models.qwen3 import Qwen3ForCausalLM

        return Qwen3ForCausalLM
    if name == "Cpm4ForCausalLM":
        from nanovllm.models.cpm4 import Cpm4ForCausalLM

        return Cpm4ForCausalLM
    if name in {"model_registry", "register_model", "get_model_class", "create_model", "list_supported_models"}:
        registry_module = importlib.import_module("nanovllm.models.model_registry")

        return getattr(registry_module, name)
    raise AttributeError(name)
