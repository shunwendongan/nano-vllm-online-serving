#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <torch/extension.h>

namespace {

constexpr int BLOCK_SIZE = 256;
constexpr int BLOCK_K = 128;

__device__ __forceinline__ float warp_reduce_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

__device__ __forceinline__ float warp_reduce_max(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_down_sync(0xffffffff, x, offset));
    }
    return x;
}

template <int BS> __device__ __forceinline__ float block_reduce_sum(float x) {
    __shared__ float warp_sums[32];
    __shared__ float block_sum;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    x = warp_reduce_sum(x);
    if (lane == 0) {
        warp_sums[wid] = x;
    }
    __syncthreads();
    x = (threadIdx.x < (BS + 31) / 32) ? warp_sums[lane] : 0.0f;
    if (wid == 0) {
        x = warp_reduce_sum(x);
        if (lane == 0) {
            block_sum = x;
        }
    }
    __syncthreads();
    return block_sum;
}

template <int BS> __device__ __forceinline__ float block_reduce_max(float x) {
    __shared__ float warp_maxes[32];
    __shared__ float block_max;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    x = warp_reduce_max(x);
    if (lane == 0) {
        warp_maxes[wid] = x;
    }
    __syncthreads();
    x = (threadIdx.x < (BS + 31) / 32) ? warp_maxes[lane] : -FLT_MAX;
    if (wid == 0) {
        x = warp_reduce_max(x);
        if (lane == 0) {
            block_max = x;
        }
    }
    __syncthreads();
    return block_max;
}

template <int BS>
__global__ void dense_mha_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                 const float* __restrict__ v, float* __restrict__ out, int B,
                                 int H, int Tq, int Tk, int D, int causal) {
    extern __shared__ float scores[];
    const int row = blockIdx.x;
    const int tq = row % Tq;
    const int h = (row / Tq) % H;
    const int b = row / (Tq * H);
    if (b >= B) {
        return;
    }
    const int q_base = (((b * H + h) * Tq + tq) * D);
    const float scale = rsqrtf((float)D);
    const int query_abs_pos = (Tk - Tq) + tq;
    float thread_max = -FLT_MAX;
    for (int tk = threadIdx.x; tk < Tk; tk += BS) {
        float score = -FLT_MAX;
        if (!causal || tk <= query_abs_pos) {
            const int k_base = (((b * H + h) * Tk + tk) * D);
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += q[q_base + d] * k[k_base + d];
            }
            score = dot * scale;
        }
        scores[tk] = score;
        thread_max = fmaxf(thread_max, score);
    }
    const float row_max = block_reduce_max<BS>(thread_max);
    __syncthreads();
    float thread_sum = 0.0f;
    for (int tk = threadIdx.x; tk < Tk; tk += BS) {
        const float p = (scores[tk] == -FLT_MAX) ? 0.0f : expf(scores[tk] - row_max);
        scores[tk] = p;
        thread_sum += p;
    }
    const float row_sum = block_reduce_sum<BS>(thread_sum);
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += BS) {
        float acc = 0.0f;
        for (int tk = 0; tk < Tk; ++tk) {
            const int v_idx = (((b * H + h) * Tk + tk) * D + d);
            acc += (scores[tk] / row_sum) * v[v_idx];
        }
        const int out_idx = (((b * H + h) * Tq + tq) * D + d);
        out[out_idx] = acc;
    }
}

template <int BS>
__global__ void gqa_mqa_kernel(const float* __restrict__ q, const float* __restrict__ k,
                               const float* __restrict__ v, float* __restrict__ out, int B,
                               int Hq, int Hkv, int Tq, int Tk, int D, int causal) {
    extern __shared__ float scores[];
    const int row = blockIdx.x;
    const int tq = row % Tq;
    const int hq = (row / Tq) % Hq;
    const int b = row / (Tq * Hq);
    if (b >= B || Hq % Hkv != 0) {
        return;
    }
    const int hkv = hq / (Hq / Hkv);
    const int q_base = (((b * Hq + hq) * Tq + tq) * D);
    const float scale = rsqrtf((float)D);
    const int query_abs_pos = (Tk - Tq) + tq;
    float thread_max = -FLT_MAX;
    for (int tk = threadIdx.x; tk < Tk; tk += BS) {
        float score = -FLT_MAX;
        if (!causal || tk <= query_abs_pos) {
            const int k_base = (((b * Hkv + hkv) * Tk + tk) * D);
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += q[q_base + d] * k[k_base + d];
            }
            score = dot * scale;
        }
        scores[tk] = score;
        thread_max = fmaxf(thread_max, score);
    }
    const float row_max = block_reduce_max<BS>(thread_max);
    __syncthreads();
    float thread_sum = 0.0f;
    for (int tk = threadIdx.x; tk < Tk; tk += BS) {
        const float p = (scores[tk] == -FLT_MAX) ? 0.0f : expf(scores[tk] - row_max);
        scores[tk] = p;
        thread_sum += p;
    }
    const float row_sum = block_reduce_sum<BS>(thread_sum);
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += BS) {
        float acc = 0.0f;
        for (int tk = 0; tk < Tk; ++tk) {
            const int v_idx = (((b * Hkv + hkv) * Tk + tk) * D + d);
            acc += (scores[tk] / row_sum) * v[v_idx];
        }
        const int out_idx = (((b * Hq + hq) * Tq + tq) * D + d);
        out[out_idx] = acc;
    }
}

template <int BS, int BK>
__global__ void streaming_gqa_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                     const float* __restrict__ v, float* __restrict__ out, int B,
                                     int Hq, int Hkv, int Tq, int Tk, int D, int causal) {
    extern __shared__ float shmem[];
    float* tile_scores = shmem;
    float* acc = shmem + BK;
    const int row = blockIdx.x;
    const int tq = row % Tq;
    const int hq = (row / Tq) % Hq;
    const int b = row / (Tq * Hq);
    if (b >= B || Hq % Hkv != 0) {
        return;
    }
    const int hkv = hq / (Hq / Hkv);
    const int q_base = (((b * Hq + hq) * Tq + tq) * D);
    const float scale = rsqrtf((float)D);
    const int query_abs_pos = (Tk - Tq) + tq;
    for (int d = threadIdx.x; d < D; d += BS) {
        acc[d] = 0.0f;
    }
    __syncthreads();
    float m = -FLT_MAX;
    float l = 0.0f;
    for (int tile_start = 0; tile_start < Tk; tile_start += BK) {
        const int tile_len = min(BK, Tk - tile_start);
        float thread_max = -FLT_MAX;
        for (int j = threadIdx.x; j < tile_len; j += BS) {
            const int tk = tile_start + j;
            float score = -FLT_MAX;
            if (!causal || tk <= query_abs_pos) {
                const int k_base = (((b * Hkv + hkv) * Tk + tk) * D);
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) {
                    dot += q[q_base + d] * k[k_base + d];
                }
                score = dot * scale;
            }
            tile_scores[j] = score;
            thread_max = fmaxf(thread_max, score);
        }
        const float tile_max = block_reduce_max<BS>(thread_max);
        const float m_new = fmaxf(m, tile_max);
        const float old_scale = (m == -FLT_MAX) ? 0.0f : expf(m - m_new);
        __syncthreads();
        float thread_sum = 0.0f;
        for (int j = threadIdx.x; j < tile_len; j += BS) {
            const float p = (tile_scores[j] == -FLT_MAX) ? 0.0f : expf(tile_scores[j] - m_new);
            tile_scores[j] = p;
            thread_sum += p;
        }
        const float tile_sum = block_reduce_sum<BS>(thread_sum);
        const float l_new = old_scale * l + tile_sum;
        __syncthreads();
        for (int d = threadIdx.x; d < D; d += BS) {
            float tile_acc = 0.0f;
            for (int j = 0; j < tile_len; ++j) {
                const int tk = tile_start + j;
                const int v_idx = (((b * Hkv + hkv) * Tk + tk) * D + d);
                tile_acc += tile_scores[j] * v[v_idx];
            }
            acc[d] = old_scale * acc[d] + tile_acc;
        }
        __syncthreads();
        m = m_new;
        l = l_new;
    }
    for (int d = threadIdx.x; d < D; d += BS) {
        const int out_idx = (((b * Hq + hq) * Tq + tq) * D + d);
        out[out_idx] = acc[d] / l;
    }
}

template <int BS>
__global__ void paged_decode_kernel(const float* __restrict__ q, const float* __restrict__ k_cache,
                                    const float* __restrict__ v_cache, const int* context_lens,
                                    const int* block_tables, float* __restrict__ out, int B,
                                    int Hq, int Hkv, int D, int block_size, int max_blocks,
                                    float scale) {
    const int row = blockIdx.x;
    const int hq = row % Hq;
    const int b = row / Hq;
    if (b >= B || Hq % Hkv != 0) {
        return;
    }
    const int hkv = hq / (Hq / Hkv);
    const int context_len = context_lens[b];
    const int q_base = ((b * Hq + hq) * D);
    float thread_max = -FLT_MAX;
    for (int tk = threadIdx.x; tk < context_len; tk += BS) {
        const int block_index = tk / block_size;
        const int block_offset = tk % block_size;
        if (block_index >= max_blocks) {
            continue;
        }
        const int physical_block = block_tables[b * max_blocks + block_index];
        if (physical_block < 0) {
            continue;
        }
        const int k_base = (((physical_block * block_size + block_offset) * Hkv + hkv) * D);
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += q[q_base + d] * k_cache[k_base + d];
        }
        thread_max = fmaxf(thread_max, dot * scale);
    }
    const float row_max = block_reduce_max<BS>(thread_max);
    float thread_sum = 0.0f;
    for (int tk = threadIdx.x; tk < context_len; tk += BS) {
        const int block_index = tk / block_size;
        const int block_offset = tk % block_size;
        if (block_index >= max_blocks) {
            continue;
        }
        const int physical_block = block_tables[b * max_blocks + block_index];
        if (physical_block < 0) {
            continue;
        }
        const int k_base = (((physical_block * block_size + block_offset) * Hkv + hkv) * D);
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += q[q_base + d] * k_cache[k_base + d];
        }
        thread_sum += expf(dot * scale - row_max);
    }
    const float row_sum = block_reduce_sum<BS>(thread_sum);
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += BS) {
        float acc = 0.0f;
        for (int tk = 0; tk < context_len; ++tk) {
            const int block_index = tk / block_size;
            const int block_offset = tk % block_size;
            if (block_index >= max_blocks) {
                continue;
            }
            const int physical_block = block_tables[b * max_blocks + block_index];
            if (physical_block < 0) {
                continue;
            }
            const int k_base = (((physical_block * block_size + block_offset) * Hkv + hkv) * D);
            const int v_base = k_base;
            float dot = 0.0f;
            for (int kd = 0; kd < D; ++kd) {
                dot += q[q_base + kd] * k_cache[k_base + kd];
            }
            const float p = expf(dot * scale - row_max) / row_sum;
            acc += p * v_cache[v_base + d];
        }
        out[q_base + d] = acc;
    }
}

void check_cuda_float(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name,
                " must be float32 in the first cuda_ext implementation");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor dense_mha_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
    check_cuda_float(q, "q");
    check_cuda_float(k, "k");
    check_cuda_float(v, "v");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be [B,H,T,D]");
    const c10::cuda::CUDAGuard device_guard(q.device());
    auto out = torch::empty_like(q);
    const int B = q.size(0), H = q.size(1), Tq = q.size(2), D = q.size(3);
    const int Tk = k.size(2);
    TORCH_CHECK(k.size(0) == B && v.size(0) == B && k.size(1) == H && v.size(1) == H);
    TORCH_CHECK(k.size(3) == D && v.size(3) == D && v.size(2) == Tk);
    dense_mha_kernel<BLOCK_SIZE><<<B * H * Tq, BLOCK_SIZE, Tk * sizeof(float),
                                   at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(), B,
        H, Tq, Tk, D, causal ? 1 : 0);
    return out;
}

torch::Tensor gqa_mqa_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
    check_cuda_float(q, "q");
    check_cuda_float(k, "k");
    check_cuda_float(v, "v");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be [B,H,T,D]");
    const c10::cuda::CUDAGuard device_guard(q.device());
    auto out = torch::empty_like(q);
    const int B = q.size(0), Hq = q.size(1), Tq = q.size(2), D = q.size(3);
    const int Hkv = k.size(1), Tk = k.size(2);
    TORCH_CHECK(Hq % Hkv == 0, "Hq must be divisible by Hkv");
    TORCH_CHECK(k.size(0) == B && v.size(0) == B && v.size(1) == Hkv);
    TORCH_CHECK(k.size(3) == D && v.size(3) == D && v.size(2) == Tk);
    gqa_mqa_kernel<BLOCK_SIZE><<<B * Hq * Tq, BLOCK_SIZE, Tk * sizeof(float),
                                 at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(), B,
        Hq, Hkv, Tq, Tk, D, causal ? 1 : 0);
    return out;
}

torch::Tensor streaming_gqa_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                    bool causal) {
    check_cuda_float(q, "q");
    check_cuda_float(k, "k");
    check_cuda_float(v, "v");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be [B,H,T,D]");
    const c10::cuda::CUDAGuard device_guard(q.device());
    auto out = torch::empty_like(q);
    const int B = q.size(0), Hq = q.size(1), Tq = q.size(2), D = q.size(3);
    const int Hkv = k.size(1), Tk = k.size(2);
    TORCH_CHECK(Hq % Hkv == 0, "Hq must be divisible by Hkv");
    TORCH_CHECK(D <= 512, "streaming_gqa_forward supports D <= 512 in this benchmark backend");
    const size_t shared_mem = (BLOCK_K + D) * sizeof(float);
    streaming_gqa_kernel<BLOCK_SIZE, BLOCK_K>
        <<<B * Hq * Tq, BLOCK_SIZE, shared_mem, at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(),
            B, Hq, Hkv, Tq, Tk, D, causal ? 1 : 0);
    return out;
}

torch::Tensor paged_decode_attention(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
                                     torch::Tensor context_lens, torch::Tensor block_tables,
                                     double scale) {
    check_cuda_float(q, "q");
    check_cuda_float(k_cache, "k_cache");
    check_cuda_float(v_cache, "v_cache");
    TORCH_CHECK(context_lens.is_cuda() && context_lens.scalar_type() == torch::kInt32);
    TORCH_CHECK(block_tables.is_cuda() && block_tables.scalar_type() == torch::kInt32);
    TORCH_CHECK(q.dim() == 3, "q must be [B,Hq,D]");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4,
                "k_cache/v_cache must be [num_blocks,block_size,Hkv,D]");
    TORCH_CHECK(block_tables.dim() == 2, "block_tables must be [B,max_blocks]");
    const c10::cuda::CUDAGuard device_guard(q.device());
    auto out = torch::empty_like(q);
    const int B = q.size(0), Hq = q.size(1), D = q.size(2);
    const int block_size = k_cache.size(1), Hkv = k_cache.size(2), max_blocks = block_tables.size(1);
    TORCH_CHECK(Hq % Hkv == 0, "Hq must be divisible by Hkv");
    TORCH_CHECK(k_cache.size(3) == D && v_cache.size(3) == D && v_cache.size(1) == block_size);
    paged_decode_kernel<BLOCK_SIZE><<<B * Hq, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(), k_cache.data_ptr<float>(), v_cache.data_ptr<float>(),
        context_lens.data_ptr<int>(), block_tables.data_ptr<int>(), out.data_ptr<float>(), B, Hq,
        Hkv, D, block_size, max_blocks, static_cast<float>(scale));
    return out;
}
