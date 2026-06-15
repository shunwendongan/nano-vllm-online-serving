#include <cuda_runtime.h>
// 需要分块，数据量太大，tile可以 32，但是未必性能好
#define TILE 16

__global__ void batched_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                      float* __restrict__ C, int BATCH, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    // 编号，索引映射，然后进行数据搬运到 共享内存上，注意tile，一个block里面计算一个tile里的元素
    int batch = blockIdx.z;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    const float* A_batch =
        A + batch * M * K; // 指针+数字，是按照元素累加，相当于A的首元素往后移动了A + batch * M * K
    // 的元素给A_batch（或者A_batch代表后面的矩阵部分），如果是batch时
    const float* B_batch = B + batch * K * N;
    float* C_batch = C + batch * M * N;

    for (int t = 0; t < K; t += TILE) {
        int a_col = t + threadIdx.x;
        int b_row = t + threadIdx.y;

        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A_batch[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] =
                0.0f; // 始终注意边界安全，处于向上取整，
                      //     可能最后的block里面的有线程多余，也就是tid是>=n的
        }

        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B_batch[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();
        // 一套并行操作下来，直接记住同步

#pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C_batch[row * N + col] = acc;
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, BATCH);

    batched_matmul_kernel<<<grid, block>>>(A, B, C, BATCH, M, N,
                                           K); // 将dim3结构体输入实际上就是默认自动挂
                                               // 入，blocks和threads
}