#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 32

// Basic version
__global__ void gemm_basic(const half* __restrict__ A,
                     const half* __restrict__ B,
                     half* __restrict__ C,
                     int M, int N, int K, float alpha, float beta) {
    int row = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    int col = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        float a = __half2float(A[row * K + k]);
        float b = __half2float(B[k * N + col]);
        sum += a * b;
    }

    float old_c = __half2float(C[row * N + col]);
    float out = alpha * sum + beta * old_c;
    C[row * N + col] = __float2half_rn(out);
}

// 2D register tiling version
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int NUM_THREADS = (BM / TM) * (BN / TN);

__global__ void gemm_tiled_2d(const half* __restrict__ A,
                              const half* __restrict__ B,
                              half* __restrict__ C,
                              int M, int N, int K,
                              float alpha, float beta) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float accum[TM][TN] = {};
    float reg_a[TM];
    float reg_b[TN];

    // Which TM x TN sub-tile this thread computes
    int thread_row = threadIdx.x / (BN / TN);
    int thread_col = threadIdx.x % (BN / TN);

    int block_row_start = blockIdx.y * BM;
    int block_col_start = blockIdx.x * BN;

    // Layout for cooperative loading of A (BM x BK per iteration)
    int a_load_col = threadIdx.x % BK;
    int a_load_row = threadIdx.x / BK;
    int a_stride = NUM_THREADS / BK;

    // Layout for cooperative loading of B (BK x BN per iteration)
    int b_load_col = threadIdx.x % BN;
    int b_load_row = threadIdx.x / BN;
    int b_stride = NUM_THREADS / BN;

    for (int bk = 0; bk < K; bk += BK) {
        // Load A tile into shared memory
        for (int offset = 0; offset < BM; offset += a_stride) {
            int row = a_load_row + offset;
            int g_row = block_row_start + row;
            int g_col = bk + a_load_col;
            As[row][a_load_col] = (g_row < M && g_col < K)
                ? __half2float(A[g_row * K + g_col]) : 0.0f;
        }

        // Load B tile into shared memory
        for (int offset = 0; offset < BK; offset += b_stride) {
            int row = b_load_row + offset;
            int g_row = bk + row;
            int g_col = block_col_start + b_load_col;
            Bs[row][b_load_col] = (g_row < K && g_col < N)
                ? __half2float(B[g_row * N + g_col]) : 0.0f;
        }

        __syncthreads();

        // Outer product accumulation from registers
        for (int dot = 0; dot < BK; dot++) {
            for (int i = 0; i < TM; i++)
                reg_a[i] = As[thread_row * TM + i][dot];
            for (int j = 0; j < TN; j++)
                reg_b[j] = Bs[dot][thread_col * TN + j];

            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    accum[i][j] += reg_a[i] * reg_b[j];
        }

        __syncthreads();
    }

    // Write results to C
    for (int i = 0; i < TM; i++) {
        int g_row = block_row_start + thread_row * TM + i;
        for (int j = 0; j < TN; j++) {
            int g_col = block_col_start + thread_col * TN + j;
            if (g_row < M && g_col < N) {
                float old_c = __half2float(C[g_row * N + g_col]);
                C[g_row * N + g_col] = __float2half_rn(alpha * accum[i][j] + beta * old_c);
            }
        }
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {
    dim3 block(NUM_THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_tiled_2d<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);

    cudaDeviceSynchronize();
}
