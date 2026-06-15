from nanovllm.models.gpt_oss_compat import raise_native_not_supported


class GptOssMoE:
    """Native gpt-oss MoE scaffold.

    The current stage exposes compatibility checks and an hf_auto serving path.
    Native MoE routing, MXFP4 loading, and expert execution are intentionally
    fail-fast until implemented and validated on a CUDA server.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("native gpt-oss MoE execution is not implemented")


class GptOssForCausalLM:
    def __init__(self, config):
        raise_native_not_supported(config)
