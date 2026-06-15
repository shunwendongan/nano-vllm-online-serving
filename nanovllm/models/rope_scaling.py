def normalize_default_rope_scaling(rope_scaling, *, model_name: str = "model"):
    if rope_scaling is None:
        return None
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        if rope_type == "default":
            return None
    raise NotImplementedError(f"unsupported {model_name} rope_scaling: {rope_scaling}")
