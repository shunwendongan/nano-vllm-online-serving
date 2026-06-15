#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void fp16_batched_matmul_kernel(const half* A, const half* B, half* C, int BATCH, int M,
                                           int N, int K) {
    __shared__ half As[TILE][TILE];
    __shared__ half Bs[TILE][TILE];

    int batch = blockIdx.z;

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int a_col = t + threadIdx.x;
        int b_row = t + threadIdx.y;

        if (batch < BATCH && row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[batch * M * K + row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }

        if (batch < BATCH && b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[batch * K * N + b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }

        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += __half2float(As[threadIdx.y][i]) * __half2float(Bs[i][threadIdx.x]);
        }

        __syncthreads();
    }

    if (batch < BATCH && row < M && col < N) {
        C[batch * M * N + row * N + col] = __float2half(acc);
    }
}

// A, B, C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, BATCH);

    fp16_batched_matmul_kernel<<<grid, block>>>(A, B, C, BATCH, M, N, K);
}