#include <torch/extension.h>

torch::Tensor dense_mha_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal);
torch::Tensor gqa_mqa_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal);
torch::Tensor streaming_gqa_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal);
torch::Tensor paged_decode_attention(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
                                     torch::Tensor context_lens, torch::Tensor block_tables,
                                     double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dense_mha_forward", &dense_mha_forward, "Naive dense MHA forward (CUDA)");
    m.def("gqa_mqa_forward", &gqa_mqa_forward, "Naive GQA/MQA forward (CUDA)");
    m.def("streaming_gqa_forward", &streaming_gqa_forward,
          "Streaming-softmax GQA/MQA forward (CUDA)");
    m.def("paged_decode_attention", &paged_decode_attention,
          "Naive paged decode attention forward (CUDA)");
}
