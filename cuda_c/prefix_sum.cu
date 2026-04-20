#include <cuda_runtime.h>

#define BLOCK_SIZE 1024
#define WARP_SIZE 32

// Warp-level inclusive scan using shuffle instructions
__device__ __forceinline__ float warp_scan(float val) {
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float temp = __shfl_up_sync(0xffffffff, val, offset);
        if (threadIdx.x % WARP_SIZE >= offset) {
            val += temp;
        }
    }
    return val;
}

// Block-level inclusive scan
__device__ float block_scan(float val, float* shared_sums) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    // 1. Each warp performs a scan
    val = warp_scan(val);

    // 2. The last thread of each warp writes its sum to shared memory
    if (lane == WARP_SIZE - 1) {
        shared_sums[wid] = val;
    }
    __syncthreads();

    // 3. Scan the warp sums (there are at most 32 warp sums per 1024-thread block)
    if (wid == 0) {
        shared_sums[lane] = warp_scan(lane < (BLOCK_SIZE / WARP_SIZE) ? shared_sums[lane] : 0.0f);
    }
    __syncthreads();

    // 4. Add the scanned warp sums back to each thread
    if (wid > 0) {
        val += shared_sums[wid - 1];
    }
    return val;
}

__global__ void scan_kernel(const float* input, float* output, float* block_sums, int N) {
    __shared__ float shared_sums[WARP_SIZE]; // Only need 32 floats for 1024 threads
    
    int gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float val = (gid < N) ? input[gid] : 0.0f;

    val = block_scan(val, shared_sums);

    if (gid < N) {
        output[gid] = val;
    }

    // Write the block's total sum for the second pass
    if (threadIdx.x == BLOCK_SIZE - 1 && block_sums != nullptr) {
        block_sums[blockIdx.x] = val;
    }
}

__global__ void apply_offsets(float* output, const float* block_sums, int N) {
    int gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (gid < N && blockIdx.x > 0) {
        output[gid] += block_sums[blockIdx.x - 1];
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float* d_block_sums = nullptr;

    if (num_blocks > 1) {
        cudaMalloc(&d_block_sums, num_blocks * sizeof(float));
    }

    // Pass 1: Local scan and collect block sums
    scan_kernel<<<num_blocks, BLOCK_SIZE>>>(input, output, d_block_sums, N);

    // Pass 2: Scan the block sums (for N=250k, this is only ~245 elements)
    if (num_blocks > 1) {
        scan_kernel<<<1, BLOCK_SIZE>>>(d_block_sums, d_block_sums, nullptr, num_blocks);

        // Pass 3: Add offsets to blocks
        apply_offsets<<<num_blocks, BLOCK_SIZE>>>(output, d_block_sums, N);
        
        cudaFree(d_block_sums);
    }
}