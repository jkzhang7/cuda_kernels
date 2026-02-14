#include <cuda_runtime.h>
#include <math.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) (call)
#endif

// Two-pass multi-head attention kernel.
// Each block handles one (query_position, head) pair.
// Pass 1: compute softmax statistics (max score and exp sum) over all key positions.
// Pass 2: recompute scores, apply softmax weights, and accumulate weighted V.
//
// Layout: Q, K, V, O are [N, d_model] row-major. Each head occupies a
// contiguous slice of dk = d_model / h columns.
//
// Shared memory layout: [q_shared (dk) | o_shared (dk + 3)]
//   o_shared is reused as a warp-reduction buffer during pass 1.
//   Requires dk >= num_warps (blockDim.x / 32).
__global__ void mha_two_pass_seqdot_kernel(const float* __restrict__ Q,
                                           const float* __restrict__ K,
                                           const float* __restrict__ V,
                                           float* __restrict__ O,
                                           int N, int d_model, int h, int dk,
                                           float scale) {
    extern __shared__ float shmem[];
    float* q_shared = shmem;
    float* o_shared = shmem + dk;

    int query_pos = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int head_offset = head_idx * dk;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = num_threads / 32;

    // Load query vector into shared memory
    for (int i = tid; i < dk; i += num_threads)
        q_shared[i] = Q[query_pos * d_model + head_offset + i];
    __syncthreads();

    // === Pass 1a: find max attention score ===
    float local_max = -INFINITY;
    for (int j = tid; j < N; j += num_threads) {
        float score = 0.0f;
        for (int d = 0; d < dk; d++)
            score += q_shared[d] * K[j * d_model + head_offset + d];
        local_max = fmaxf(local_max, score * scale);
    }

    // Warp-level max reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    if (lane_id == 0)
        o_shared[warp_id] = local_max;
    __syncthreads();

    // Cross-warp max reduction
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? o_shared[lane_id] : -INFINITY;
        for (int offset = 16; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        if (lane_id == 0)
            o_shared[0] = val;
    }
    __syncthreads();
    float global_max = o_shared[0];
    __syncthreads();

    // === Pass 1b: compute sum of exp(score - max) ===
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += num_threads) {
        float score = 0.0f;
        for (int d = 0; d < dk; d++)
            score += q_shared[d] * K[j * d_model + head_offset + d];
        local_sum += expf(score * scale - global_max);
    }

    // Warp-level sum reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    if (lane_id == 0)
        o_shared[warp_id] = local_sum;
    __syncthreads();

    // Cross-warp sum reduction
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? o_shared[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (lane_id == 0)
            o_shared[0] = val;
    }
    __syncthreads();
    float global_sum = o_shared[0];
    __syncthreads();

    // === Pass 2: accumulate softmax-weighted V ===
    for (int i = tid; i < dk; i += num_threads)
        o_shared[i] = 0.0f;
    __syncthreads();

    for (int j = tid; j < N; j += num_threads) {
        float score = 0.0f;
        for (int d = 0; d < dk; d++)
            score += q_shared[d] * K[j * d_model + head_offset + d];
        float weight = expf(score * scale - global_max) / global_sum;

        for (int d = 0; d < dk; d++)
            atomicAdd(&o_shared[d], weight * V[j * d_model + head_offset + d]);
    }
    __syncthreads();

    // Write output
    for (int i = tid; i < dk; i += num_threads)
        O[query_pos * d_model + head_offset + i] = o_shared[i];
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int N, int d_model, int h) {
    if (!Q || !K || !V || !output) return;
    if (N <= 0 || d_model <= 0 || h <= 0) return;
    if (d_model % h != 0) return;

    int dk = d_model / h;
    float scale = 1.0f / sqrtf((float)dk);

    int threads = 256;

    dim3 grid((unsigned)N, (unsigned)h, 1);
    dim3 block((unsigned)threads, 1, 1);

    size_t shmem = (size_t)(2 * dk + 3) * sizeof(float);

    mha_two_pass_seqdot_kernel<<<grid, block, shmem>>>(Q, K, V, output, N, d_model, h, dk, scale);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}
