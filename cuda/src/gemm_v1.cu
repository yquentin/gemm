template <int BLOCK>
__global__ void kernel_v1(int m, int n, int k, const float *a, int lda,
        const float *b, int ldb, float *c, int ldc) {
    __shared__ float tile_A[BLOCK][BLOCK];
    __shared__ float tile_B[BLOCK][BLOCK];

    float sum = 0.0f;
    for (int _k = 0; _k < k; _k += BLOCK) {
        // load A and B
        int A_off = (BLOCK * blockIdx.y + threadIdx.y) * lda + _k + threadIdx.x;
        int B_off = (_k + threadIdx.y) * ldb + BLOCK * blockIdx.x + threadIdx.x;
        tile_A[threadIdx.y][threadIdx.x] = *(a + A_off);
        tile_B[threadIdx.y][threadIdx.x] = *(b + B_off);
        __syncthreads(); // wait all threads in a block to finish copy

        // compute gemm on tile
        for (int i = 0; i < BLOCK; i++) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }
        __syncthreads(); // wait all threads to finish compute
    }

    // store C
    int C_off = (BLOCK * blockIdx.y + threadIdx.y) * ldc + blockIdx.x * BLOCK
            + threadIdx.x;
    *(c + C_off) = sum;
}

void gemm_v1(const float *A, const float *B, float *C, int M, int K, int N) {
    const int BLOCK = 32;
    dim3 dimBlock(BLOCK, BLOCK);
    dim3 dimGrid(M / BLOCK, N / BLOCK);
    kernel_v1<BLOCK><<<dimGrid, dimBlock>>>(M, N, K, A, K, B, N, C, N);
}