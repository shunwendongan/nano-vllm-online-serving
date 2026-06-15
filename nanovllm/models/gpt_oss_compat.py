from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class GptOssCompatibilityReport:
    is_gpt_oss: bool
    model_id_hint: str
    model_type: str | None
    architectures: list[str]
    num_hidden_layers: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None
    num_experts: int | None
    num_experts_per_tok: int | None
    uses_moe: bool
    quantization_hint: str | None
    native_supported: bool
    unsupported_reasons: list[str]

    def to_dict(self):
        return asdict(self)


def _config_to_dict(config_or_path: Any):
    if isinstance(config_or_path, dict):
        return dict(config_or_path)
    if isinstance(config_or_path, str):
        config_path = os.path.join(config_or_path, "config.json")
        if os.path.isdir(config_or_path) and os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        return {"_model_id_hint": config_or_path}
    result = {}
    for name in dir(config_or_path):
        if name.startswith("_"):
            continue
        try:
            value = getattr(config_or_path, name)
        except Exception:
            continue
        if isinstance(value, (str, int, float, bool, list, tuple, dict, type(None))):
            result[name] = value
    return result


def _as_int(config: dict, *names: str):
    for name in names:
        value = config.get(name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def inspect_gpt_oss_config(config_or_path: Any) -> GptOssCompatibilityReport:
    config = _config_to_dict(config_or_path)
    model_type = config.get("model_type")
    architectures = list(config.get("architectures") or [])
    model_id_hint = str(
        config.get("_name_or_path")
        or config.get("_model_id_hint")
        or config.get("name_or_path")
        or ""
    )
    haystack = " ".join([model_id_hint, str(model_type or ""), *architectures]).lower()
    is_gpt_oss = "gpt-oss" in haystack or "gpt_oss" in haystack or model_type in {"gpt_oss", "gpt-oss"}
    num_experts = _as_int(config, "num_experts", "n_experts", "num_local_experts")
    num_experts_per_tok = _as_int(
        config,
        "num_experts_per_tok",
        "num_experts_per_token",
        "moe_top_k",
        "router_top_k",
    )
    quantization_config = config.get("quantization_config") or {}
    quantization_hint = (
        str(quantization_config.get("quant_method") or quantization_config.get("format"))
        if isinstance(quantization_config, dict) and quantization_config
        else None
    )
    uses_moe = bool(num_experts and num_experts > 1)
    unsupported_reasons = []
    if is_gpt_oss:
        unsupported_reasons.append("native gpt-oss path is not implemented yet")
    if uses_moe:
        unsupported_reasons.append("MoE router/expert execution is not implemented in native path")
    if quantization_hint and "mxfp4" in quantization_hint.lower():
        unsupported_reasons.append("MXFP4 native weight loading/dequantization is not implemented")
    if not unsupported_reasons and is_gpt_oss:
        unsupported_reasons.append("gpt-oss config detected but native compatibility has not been proven")
    return GptOssCompatibilityReport(
        is_gpt_oss=is_gpt_oss,
        model_id_hint=model_id_hint,
        model_type=model_type,
        architectures=architectures,
        num_hidden_layers=_as_int(config, "num_hidden_layers", "n_layer", "n_layers"),
        num_attention_heads=_as_int(config, "num_attention_heads", "n_head", "n_heads"),
        num_key_value_heads=_as_int(config, "num_key_value_heads", "n_kv_heads"),
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        uses_moe=uses_moe,
        quantization_hint=quantization_hint,
        native_supported=False if is_gpt_oss else True,
        unsupported_reasons=unsupported_reasons,
    )


def is_gpt_oss_config(config_or_path: Any) -> bool:
    return inspect_gpt_oss_config(config_or_path).is_gpt_oss


class GptOssNativeNotSupportedError(NotImplementedError):
    pass


def raise_native_not_supported(config_or_path: Any):
    report = inspect_gpt_oss_config(config_or_path)
    raise GptOssNativeNotSupportedError(
        "gpt-oss native execution is not implemented in nano-vLLM yet; "
        "start the server with --model-backend hf_auto for current-stage smoke validation. "
        f"compatibility_report={report.to_dict()}"
    )
