#include <cuda_runtime.h>

// RMSNorm over one 1D vector:
// output[i] = input[i] * gamma / sqrt(mean(input^2) + eps) + beta
//
// One block is enough for N <= 100000. 256 threads cooperatively reduce the
// sum of squares, then all threads write the normalized output.
__global__ void rms_norm_kernel(const float* __restrict__ input, float* __restrict__ output, int N,
                                float gamma, float beta, float eps) {
    __shared__ float partial[256];
    __shared__ float inv_rms;

    int tid = threadIdx.x;

    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float x = input[i];
        sum_sq += x * x;
    } // 这里是用一个block，不需要blockidx和全局idx（n小于10000，一个block即可）

    partial[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads(); //
    }

    if (tid == 0) {
        float mean_sq = partial[0] / static_cast<float>(N);
        inv_rms = rsqrtf(mean_sq + eps); // 全局依赖必须要先算出来
    }
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = input[i] * inv_rms * gamma + beta;
    }
}

// input and output are device pointers.
extern "C" void solve(const float* input, float* output, int N, float gamma, float beta,
                      float eps) {
    rms_norm_kernel<<<1, 256>>>(input, output, N, gamma, beta, eps);
}
