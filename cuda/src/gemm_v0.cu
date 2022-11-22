template <int BLOCK>
__global__ void kernel_v0(int m, int n, int k, const float *a, int lda,
        const float *b, int ldb, float *c, int ldc) {
    int _n = blockIdx.x * BLOCK + threadIdx.x;
    int _m = blockIdx.y * BLOCK + threadIdx.y;

    float sum = 0.f;
    for (int i = 0; i < k; ++i) {
        sum += a[_m * k + i] * b[i * n + _n];
    }
    c[_m * n + _n] = sum;
}

void gemm_v0(const float *A, const float *B, float *C, int M, int K, int N) {
    const int BLOCK = 32;
    dim3 dimBlock(BLOCK, BLOCK);
    dim3 dimGrid(M / BLOCK, N / BLOCK);
    kernel_v0<BLOCK><<<dimGrid, dimBlock>>>(M, N, K, A, K, B, N, C, N);
}