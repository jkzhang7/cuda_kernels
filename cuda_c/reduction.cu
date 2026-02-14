#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int BLOCKS_PER_GRID = 1024;
constexpr int THREADS_PER_GRID = THREADS_PER_BLOCK * BLOCKS_PER_GRID;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduction_kernel(const float* input, float* output, int N) {
    __shared__ float shmem[WARPS_PER_BLOCK];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    // Grid-stride loop to accumulate per-thread sum
    float sum = 0.0f;
    for (int i = tid; i < N; i += THREADS_PER_GRID) {
        sum += input[i];
    }

    // First reduction: within each warp
    sum = warp_reduce_sum(sum);

    if (lane_id == 0) {
        shmem[warp_id] = sum;
    }

    __syncthreads();

    // Second reduction: first warp reduces across all warps in this block
    if (warp_id == 0) {
        sum = (lane_id < WARPS_PER_BLOCK) ? shmem[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);

        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    reduction_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(input, output, N);
}
