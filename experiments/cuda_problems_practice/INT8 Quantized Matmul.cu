#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define TILE 16

__device__ __forceinline__ int8_t clamp_int8(int x) { // 编译器直接展开内联函数，降低开销
    if (x > 127)
        return 127;
    if (x < -128)
        return -128;
    return static_cast<int8_t>(x);
}

__global__ void int8_quantized_matmul_kernel(const int8_t* A, const int8_t* B, int8_t* C, int M,
                                             int N, int K, float scale_A, float scale_B,
                                             float scale_C, int zero_point_A, int zero_point_B,
                                             int zero_point_C) {
    __shared__ int8_t As[TILE][TILE];
    __shared__ int8_t Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    int acc = 0;

    for (int t = 0; t < K; t += TILE) {
        int a_col = t + threadIdx.x;
        int b_row = t + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : static_cast<int8_t>(zero_point_A);
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : static_cast<int8_t>(zero_point_B);

        __syncthreads(); // 数据转移到共享内存，

#pragma unroll
        for (int i = 0; i < TILE; ++i) {
            int a = static_cast<int>(As[threadIdx.y][i]) - zero_point_A;
            int b = static_cast<int>(Bs[i][threadIdx.x]) - zero_point_B;
            acc += a * b;
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float scaled = static_cast<float>(acc) * scale_A * scale_B / scale_C;
        int q = static_cast<int>(roundf(scaled)) + zero_point_C;
        C[row * N + col] = clamp_int8(q);
    }
}

// Computes quantized C for A(MxK) @ B(KxN), all matrices stored row-major.
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C) {
    dim3 block(TILE, TILE); // tile version,erase pressure from data transform
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE); // notice! x stand for row

    int8_quantized_matmul_kernel<<<grid, block>>>(A, B, C, M, N, K, scale_A, scale_B, scale_C,
                                                  zero_point_A, zero_point_B, zero_point_C);
}
