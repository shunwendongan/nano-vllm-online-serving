#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// 手写的 MHA 前向算子。
//
// 输入张量假设已经是 Q/K/V 投影后的结果：
//   q:   [B, H, Tq, D]
//   k:   [B, H, Tk, D]
//   v:   [B, H, Tk, D]  // B=batch，大头数，Tk=key/value 序列长度，D=head 维度
//   out: [B, H, Tq, D]
//
// 张量在内存中按行优先连续存储：
//   index = (((b * H + h) * T + t) * D + d)
//
// 这是一个面试级别的简洁实现：
//   1) 一个 CUDA block 计算一条注意力输出行：(b, h, tq)
//   2) logits/概率存储在共享内存 scores 中，长度为 Tk
//   3) softmax 采用减去行最大值的数值稳定化做法
//   4) 因果屏蔽兼容自注意力和 KV-cache decode 的因果屏蔽
//
// Launch 要求：
//   共享内存字节数 shared_mem_bytes 必须 >= Tk * sizeof(float)

// warp 内归约求和（假设一个 warp 为 32 个线程）
// 说明：使用 CUDA 原语 __shfl_down_sync 在同一个 warp 内做树形归约，
// 从 offset=16 开始，每轮将相距 offset 的线程值相加，逐步把 32 个线程折半。
// 经过 16、8、4、2、1 的 offset 后，lane0 上会聚合整个 warp 的总和。
// 返回值：每个线程都会得到该 warp 的总和，便于后续把 warp 结果写入共享内存。
static __device__ __forceinline__ float warp_reduce_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(
            0xffffffff, x,
            offset); // warp内数据交换，这个shfl命令是读取当前lane_id+offset的线程的值
    }
    return x;
}

// warp 内归约求最大值。原理同 warp_reduce_sum 相同，用于 softmax 中求当前 warp 的最大值。
static __device__ __forceinline__ float warp_reduce_max(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x,
                  __shfl_down_sync(0xffffffff, x,
                                   offset)); // 0xffffffff 表示当前 warp 的 32 个线程都参与掩码计算
    }
    return x;
}

// block 级别的归约求和（适用于整个 CUDA block）
// 先在每个 warp 内归约（warp_reduce_sum），再把每个 warp 的结果写到共享内存
// 最后由第 0 个 warp 对这些 warp 结果再归约得到整个 block 的和。
template <int BLOCK_SIZE> static __device__ __forceinline__ float block_reduce_sum(float x) {
    __shared__ float warp_sums[32];
    __shared__ float block_sum;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    x = warp_reduce_sum(x);
    if (lane == 0) {
        warp_sums[wid] = x;
    }
    __syncthreads();

    x = (threadIdx.x < (BLOCK_SIZE + 31) / 32) ? warp_sums[lane] : 0.0f;
    if (wid == 0) {
        x = warp_reduce_sum(x);
        if (lane == 0) {
            block_sum = x;
        }
    }
    __syncthreads();
    return block_sum;
}

// block 级别的归约求最大值，逻辑与 block_reduce_sum 相似，但用最大值代替和。
template <int BLOCK_SIZE> static __device__ __forceinline__ float block_reduce_max(float x) {
    __shared__ float warp_maxes[32];
    __shared__ float block_max;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    x = warp_reduce_max(x);
    if (lane == 0) {
        warp_maxes[wid] = x;
    }
    __syncthreads();

    x = (threadIdx.x < (BLOCK_SIZE + 31) / 32) ? warp_maxes[lane] : -FLT_MAX;
    if (wid == 0) {
        x = warp_reduce_max(x);
        if (lane == 0) {
            block_max = x;
        }
    }
    __syncthreads();
    return block_max;
}

// 单头 MHA 的简洁实现：每个 CUDA block 计算一个注意力输出行 (b, h, tq)
// 输入假设已经是 Q, K, V 投影后的结果，layout: [B, H, T, D]
// 实现要点：
//  1) 每个 block 在共享内存中分配一个长度为 Tk 的 scores 数组，存储 logits 或 softmax 分子。
//  2) 先计算 score = (q·k)/sqrt(D)，并根据 causal mask 将不可见位置设为 -INF。
//  3) softmax 先减去 row_max，保证数值稳定，再归一化。
//  4) 用归一化权重对 V 做加权求和得到输出。
template <int BLOCK_SIZE>
__global__ void mha_forward_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                   const float* __restrict__ v, float* __restrict__ out, int B,
                                   int H, int Tq, int Tk, int D,
                                   int causal) { // 一般 Tk 与 Tq 相等，但 decode 模式下 Tk 可能更大
    // scores 存放每个 tk 对应的 logit 或 softmax 分子，大小为 Tk * sizeof(float)
    extern __shared__ float scores[];

    // 这里把 blockIdx.x 一维展开为 batch/head/query 三个维度：
    //   row = b * H * Tq + h * Tq + tq
    // 其中：
    //   tq = row % Tq          // query 位置
    //   h  = (row / Tq) % H    // head 编号
    //   b  = row / (Tq * H)    // batch 编号
    const int row = blockIdx.x;
    const int tq = row % Tq;
    const int h = (row / Tq) % H;
    const int b = row / (Tq * H);

    if (b >= B) {
        return;
    }

    // q_base 指向当前 (b,h,tq) 的 Q 向量起始地址
    const int q_base = (((b * H + h) * Tq + tq) * D);
    // 缩放因子 1/sqrt(D)，避免内积过大导致 softmax 数值不稳定
    const float scale = rsqrtf((float)D);

    // 当 Tq < Tk 时，这里把 query 位置对齐到 KV cache 的尾部。
    // 例如 decode 步骤：Tq = 1, Tk = cached_length，则 query_abs_pos = Tk - 1。
    const int query_abs_pos = (Tk - Tq) + tq;

    // thread_max 记录当前线程在负责的 tk 范围内的最大 logit（用于 softmax 的数值稳定化）
    float thread_max = -FLT_MAX;

    for (int tk = threadIdx.x; tk < Tk; tk += BLOCK_SIZE) {
        float score = -FLT_MAX;
        // 因果屏蔽：当 causal==1 时只允许访问小于等于 query_abs_pos 的 tk
        const bool visible = (causal == 0) || (tk <= query_abs_pos);

        if (visible) {
            const int k_base = (((b * H + h) * Tk + tk) * D);
            float dot = 0.0f;

            // 计算 q·k（长度 D 的向量内积）
            for (int d = 0; d < D; ++d) {
                dot += q[q_base + d] * k[k_base + d];
            }
            score = dot * scale;
        }

        // 不可见的位置保留 -FLT_MAX 表示概率为 0
        scores[tk] = score;
        thread_max = fmaxf(thread_max, score);
    }

    // 全 block 求 max，用于 softmax 的数值稳定化（减去 row_max）
    const float row_max = block_reduce_max<BLOCK_SIZE>(thread_max);
    __syncthreads();

    // 计算 softmax 的分子并累加每个线程负责的部分求和
    float thread_sum = 0.0f;
    for (int tk = threadIdx.x; tk < Tk; tk += BLOCK_SIZE) {
        const float p = (scores[tk] == -FLT_MAX) ? 0.0f : expf(scores[tk] - row_max);
        scores[tk] = p; // 先把分子写回共享内存
        thread_sum += p;
    }

    // 全 block 求和得到 softmax 的分母
    const float row_sum = block_reduce_sum<BLOCK_SIZE>(thread_sum);
    __syncthreads();

    for (int d = threadIdx.x; d < D; d += BLOCK_SIZE) {
        float acc = 0.0f;

        // 对所有 kv 时间步做加权和（注意 scores 存的是分子，需要除以 row_sum）
        for (int tk = 0; tk < Tk; ++tk) {
            const int v_idx = (((b * H + h) * Tk + tk) * D + d);
            acc += (scores[tk] / row_sum) * v[v_idx];
        }

        const int out_idx = (((b * H + h) * Tq + tq) * D + d);
        out[out_idx] = acc; // 写回输出
    }
}

extern "C" void mha_forward_cuda(const float* q, const float* k, const float* v, float* out, int B,
                                 int H, int Tq, int Tk, int D, int causal, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    const int grid = B * H * Tq;
    const size_t shared_mem = (size_t)Tk * sizeof(float);

    mha_forward_kernel<BLOCK_SIZE>
        <<<grid, BLOCK_SIZE, shared_mem, stream>>>(q, k, v, out, B, H, Tq, Tk, D, causal);
}

// 变体 1：GQA / MQA 前向计算。
//
// q:   [B, Hq,  Tq, D]
// k:   [B, Hkv, Tk, D]
// v:   [B, Hkv, Tk, D]
// out: [B, Hq,  Tq, D]
//
// 映射规则：
//   group_size = Hq / Hkv
//   hkv = hq / group_size
//
// 当 Hkv == Hq 时，退化为标准 MHA。
// 当 Hkv == 1 时，退化为 MQA。
// 否则就是 GQA。
// GQA / MQA 变体的参考实现：
// - q: [B, Hq, Tq, D]
// - k,v: [B, Hkv, Tk, D]
// - out: [B, Hq, Tq, D]
// 映射规则：group_size = Hq / Hkv，hkv = hq / group_size
// 目的是把多个 query head 映射到更少的 kv head，从而共用 KV cache，
// 降低缓存占用与全局内存带宽。代码逻辑与单头 MHA 类似，只是索引中使用了 hq/hkv 映射。
template <int BLOCK_SIZE>
__global__ void gqa_mqa_forward_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                       const float* __restrict__ v, float* __restrict__ out, int B,
                                       int Hq, int Hkv, int Tq, int Tk, int D, int causal) {
    // scores 与单头实现相同：共享内存中存储每个 tk 对应的 logit 或 softmax 分子
    extern __shared__ float scores[];

    const int row = blockIdx.x;
    const int tq = row % Tq;
    const int hq = (row / Tq) % Hq;
    const int b = row / (Tq * Hq);

    if (b >= B || Hq % Hkv != 0) {
        return; // 参数不合法或 batch 越界
    }

    // group_size: 每个 kv head 服务的 query head 数量
    const int group_size = Hq / Hkv;
    // 把 query head 映射到 kv head
    const int hkv = hq / group_size;
    const int q_base = (((b * Hq + hq) * Tq + tq) * D);
    const float scale = rsqrtf((float)D);
    const int query_abs_pos = (Tk - Tq) + tq;

    float thread_max = -FLT_MAX;

    // 与单头实现相同：计算 q·k 并写入 scores，共享 causal 逻辑
    for (int tk = threadIdx.x; tk < Tk; tk += BLOCK_SIZE) {
        float score = -FLT_MAX;
        const bool visible = (causal == 0) || (tk <= query_abs_pos);

        if (visible) {
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

    const float row_max = block_reduce_max<BLOCK_SIZE>(thread_max);
    __syncthreads();

    float thread_sum = 0.0f;
    for (int tk = threadIdx.x; tk < Tk; tk += BLOCK_SIZE) {
        const float p = (scores[tk] == -FLT_MAX) ? 0.0f : expf(scores[tk] - row_max);
        scores[tk] = p;
        thread_sum += p;
    }

    const float row_sum = block_reduce_sum<BLOCK_SIZE>(thread_sum);
    __syncthreads();

    for (int d = threadIdx.x; d < D; d += BLOCK_SIZE) {
        float acc = 0.0f;

        // 注意这里 v 的索引使用的是 hkv 而不是 hq（共享 KV）
        for (int tk = 0; tk < Tk; ++tk) {
            const int v_idx = (((b * Hkv + hkv) * Tk + tk) * D + d);
            acc += (scores[tk] / row_sum) * v[v_idx];
        }

        const int out_idx = (((b * Hq + hq) * Tq + tq) * D + d);
        out[out_idx] = acc;
    }
}

extern "C" void gqa_mqa_forward_cuda(const float* q, const float* k, const float* v, float* out,
                                     int B, int Hq, int Hkv, int Tq, int Tk, int D, int causal,
                                     cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    const int grid = B * Hq * Tq;
    const size_t shared_mem = (size_t)Tk * sizeof(float);

    gqa_mqa_forward_kernel<BLOCK_SIZE>
        <<<grid, BLOCK_SIZE, shared_mem, stream>>>(q, k, v, out, B, Hq, Hkv, Tq, Tk, D, causal);
}

// 变体 2：FlashAttention 风格的 streaming GQA/MQA/MHA 前向计算。
//
// 这是精确注意力，不是近似算法：
//   - 将 K/V 按块流式读入
//   - 保持在线 softmax 状态 m、l
//   - 不会显式构建完整的 [Tq, Tk] attention score 矩阵
//
// 这仍然是手写的面试级 kernel，而不是生产级 FA2/FA3 kernel。
// 生产级 kernel 会对多条 Q 行做 tile，向量化加载，使用 half/bfloat16，
// 使用张量核，并精细调度共享内存。
template <int BLOCK_SIZE, int BLOCK_K>
__global__ void flash_gqa_forward_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                         const float* __restrict__ v, float* __restrict__ out,
                                         int B, int Hq, int Hkv, int Tq, int Tk, int D,
                                         int causal) {
    // 共享内存布局说明：
    // - tile_scores: 用来存放当前 tile 的 logits（长度 BLOCK_K）
    // - acc: 用来累积加权后的 V（长度 D），每个线程按 stride 更新
    // 共享内存大小在外部 wrapper 由 (BLOCK_K + D) * sizeof(float) 指定
    extern __shared__ float shmem[];
    float* tile_scores = shmem;   // [BLOCK_K]
    float* acc = shmem + BLOCK_K; // [D]

    const int row = blockIdx.x;
    const int tq = row % Tq;
    const int hq = (row / Tq) % Hq;
    const int b = row / (Tq * Hq);

    if (b >= B || Hq % Hkv != 0) {
        return;
    }

    const int group_size = Hq / Hkv;
    const int hkv = hq / group_size;
    const int q_base = (((b * Hq + hq) * Tq + tq) * D);
    const float scale = rsqrtf((float)D);
    const int query_abs_pos = (Tk - Tq) + tq;

    // 初始化累加器：acc 保存被归一化前的加权和（未除以 l）
    for (int d = threadIdx.x; d < D; d += BLOCK_SIZE) {
        acc[d] = 0.0f;
    }
    __syncthreads();

    // online softmax 状态变量：
    // m: 当前已处理 tile 的全局最大 logit（用于数值稳定化）
    // l: 当前已处理 tile 的归一化分母（累积的 exp 部分，已按 m 标准化）
    // 初始时 m = -inf, l = 0
    float m = -FLT_MAX;
    float l = 0.0f;

    for (int tile_start = 0; tile_start < Tk; tile_start += BLOCK_K) {
        const int remain = Tk - tile_start;
        const int tile_len = (remain < BLOCK_K) ? remain : BLOCK_K;
        float thread_max = -FLT_MAX;

        for (int j = threadIdx.x; j < tile_len; j += BLOCK_SIZE) {
            const int tk = tile_start + j;
            float score = -FLT_MAX;
            const bool visible = (causal == 0) || (tk <= query_abs_pos);

            if (visible) {
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

        const float tile_max = block_reduce_max<BLOCK_SIZE>(thread_max);
        __syncthreads();

        // online softmax 的关键更新：
        // 设当前全局（到上一个 tile）状态为 (m, l)，当前 tile 的局部最大为 tile_max。
        // 1) 计算新的全局最大 m_new = max(m, tile_max)
        // 2) 把旧的累积值按比例缩放到新的基准：old_scale = exp(m - m_new)
        //    当 m == -FLT_MAX（初始）时，old_scale 定义为 0
        // 3) 本 tile 的分子按 m_new 进行偏移并 exponentiate：p_j = exp(tile_scores[j] - m_new)
        // 4) 更新分母 l_new = old_scale * l + sum_j p_j
        // 5) 更新累加器 acc[d] = old_scale * acc[d] + sum_j p_j * v_j
        // 这样我们始终保持 acc 与 l 在同一个数值基准 m_new 下，保证数值稳定。
        const float m_new = fmaxf(m, tile_max);
        const float old_scale = (m == -FLT_MAX) ? 0.0f : expf(m - m_new);

        float thread_sum = 0.0f;
        for (int j = threadIdx.x; j < tile_len; j += BLOCK_SIZE) {
            const float p = (tile_scores[j] == -FLT_MAX) ? 0.0f : expf(tile_scores[j] - m_new);
            tile_scores[j] = p; // 重用 tile_scores 存放分子
            thread_sum += p;
        }

        // 求本 tile 的分子和
        const float tile_sum = block_reduce_sum<BLOCK_SIZE>(thread_sum);
        const float l_new = old_scale * l + tile_sum;
        __syncthreads();

        // 更新 acc：对该 tile 做加权和，然后乘以相应缩放并加到旧 acc
        for (int d = threadIdx.x; d < D; d += BLOCK_SIZE) {
            float tile_acc = 0.0f;

            for (int j = 0; j < tile_len; ++j) {
                const int tk = tile_start + j;
                const int v_idx = (((b * Hkv + hkv) * Tk + tk) * D + d);
                tile_acc += tile_scores[j] * v[v_idx];
            }

            // 把旧 acc 按 old_scale 缩放后加上当前 tile 的贡献
            acc[d] = old_scale * acc[d] + tile_acc;
        }
        __syncthreads();

        // 把在线状态替换为新的全局状态
        m = m_new;
        l = l_new;
    }

    for (int d = threadIdx.x; d < D; d += BLOCK_SIZE) {
        const int out_idx = (((b * Hq + hq) * Tq + tq) * D + d);
        out[out_idx] = acc[d] / l;
    }
}

extern "C" void flash_gqa_forward_cuda(const float* q, const float* k, const float* v, float* out,
                                       int B, int Hq, int Hkv, int Tq, int Tk, int D, int causal,
                                       cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int BLOCK_K = 128;
    const int grid = B * Hq * Tq;
    const size_t shared_mem = (size_t)(BLOCK_K + D) * sizeof(float);

    flash_gqa_forward_kernel<BLOCK_SIZE, BLOCK_K>
        <<<grid, BLOCK_SIZE, shared_mem, stream>>>(q, k, v, out, B, Hq, Hkv, Tq, Tk, D, causal);
}

/*
================================================================================
 面试题式详细说明（放在文件尾部，便于阅读）：

 1) GQA / MQA 的 head 映射与为什么能减少 KV cache 显存和带宽
 --------------------------------------------------------------------------------
 概念回顾：
 - Hq：query head 的数量，即模型在 query 方向拆分出的头数
 - Hkv：key/value head 的数量，即模型在 KV 方向拆分出的头数

 映射规则（本文件实现使用）：
   group_size = Hq / Hkv
   hkv = hq / group_size   // 将 query head 映射到对应的 kv head

 解释（直观）：
 - 当 Hkv < Hq（例如 Hq=64, Hkv=16）时，group_size = 4，意味着每个 KV head 被 4 个 Q head 共享。
 - 物理上 KV cache 存储的是每个 head 的 K/V 向量，若 Hkv < Hq，则总的 KV cache 大小为
     B * Hkv * Tk * D * sizeof(dtype)
   而不是 B * Hq * Tk * D * sizeof(dtype)。因此显存占用按比例减少（缩小为原来的 Hkv/Hq）。

 为什么能降低带宽：
 - 在推理（decode）或批量推理中，query 会重复访问 KV cache 以计算注意力。若每个 query head 都有独立
   KV，则每个 head 的计算都要读一遍对应的 KV 数据；
 - 当多个 query head 共享同一个 kv head 时，GPU 上的缓存行/全局内存访问可以复用同一份 KV
数据，减少对 全局内存的重复读取；
 - 从带宽角度看，读取 KV 的次数减少了约 group_size 倍（理想情况下），从而降低内存带宽压力。

 设计权衡：
 - 好处：显著减少 KV cache 占用和内存带宽；在多头数目很大时尤其有效。
 - 坏处：每个 kv head 必须同时服务多个 query head，模型表达能力可能受限（不同 query heads
无法拥有完全 独立的 KV 表征）；这通常通过在投影层调整参数（例如不同的线性投影）来缓解。

 实践要点：
 - Hq 必须能被 Hkv 整除（Hq % Hkv == 0），否则映射不均匀；实现中已检查该条件并返回错误。
 - 在实际工程中 GQA/MQA 的权重矩阵会做对应的 reshape/共享，使得向量维度对齐且性能友好。

 --------------------------------------------------------------------------------
 2) FlashAttention 风格的 streaming 在线 softmax 的 m/l/acc 更新，以及为什么不需要落地完整 [Tq, Tk]
attention 矩阵
 --------------------------------------------------------------------------------
 背景：标准实现需要先计算完整的 attention logits 矩阵 A，形状为 [Tq, Tk]，然后对每行做
softmax，再乘以 V， 这会产生很大的内存和带宽开销（O(Tq*Tk)）。

 FlashAttention 的核心思想（本文件的 streaming 实现中）：
 - 将 K/V 按时间维度分成若干块（tile，大约长度 BLOCK_K），一次只读入一个 tile 到共享内存；
 - 对每个 query（或一小组 query）维护在线的归一化状态 (m, l) 和累加器 acc，逐块更新；
 - 从数值上等价于先计算完整 logits 再做 softmax，但只需要 O(D + BLOCK_K) 的额外内存，而非 O(Tk) 或
O(Tq*Tk) 的矩阵。

 关键变量含义与更新公式（与代码一致）：
 - 对于某一条 query 行，设已经处理过前若干 tiles 的在线状态为 (m, l, acc)：
     m：当前已处理部分的最大 logit，用于数值稳定化；
     l：当前已处理部分对应的分母，已按 m 标准化的 exp 累加；
     acc[d]：当前已处理部分的 V 加权和，按 m 标准化。

 - 处理新 tile 时，假设 tile 内每个位置 j 的原始 logit 为 s_j（未缩放）：
     tile_max = max_j s_j
     m_new = max(m, tile_max)
     old_scale = exp(m - m_new)    // 当 m=-inf 时定义为 0

 - 对 tile 内每个位置计算 p_j = exp(s_j - m_new)（以 m_new 为基准的分子），
   计算 tile_sum = sum_j p_j

 - 更新分母：
     l_new = old_scale * l + tile_sum

 - 更新加权和（每个维度 d）：
     tile_acc[d] = sum_j p_j * v_j[d]
     acc_new[d] = old_scale * acc[d] + tile_acc[d]

 - 最终输出（全部 tile 处理完后）：
     out[d] = acc[d] / l

 说明：
 - 该更新过程等价于把全部 logits 统一按最终全局最大值 M 进行规范化后求和，
   但通过逐步把旧的基准从 m 切换到 m_new（并乘以 old_scale）实现在线重基准化；
 - 由于每次只需保存标量 m、l 和长度为 D 的 acc（以及当前 tile 的临时 scores），
   所以内存需求从 O(Tk) 或 O(Tq*Tk) 降到 O(D + BLOCK_K)。
*/
// 可选的 LeetGPU 风格 wrapper 名称。若平台以不同方式提供 B/H/T/D，请调整该函数签名。
extern "C" void solve(const float* q, const float* k, const float* v, float* out, int B, int H,
                      int Tq, int Tk, int D, int causal) {
    mha_forward_cuda(q, k, v, out, B, H, Tq, Tk, D, causal, 0);
}
