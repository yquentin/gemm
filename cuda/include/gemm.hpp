// native
void gemm_v0(const float *A, const float *B, float *C, int M, int K, int N);

// block to use shared memory
void gemm_v1(const float *A, const float *B, float *C, int M, int K, int N);

// increase the block size to 128x128, and compute a 4x4 tile per thread to reduce thread number per block
void gemm_v2(const float *A, const float *B, float *C, int M, int K, int N);

// use 8x8 tile to reuse more data in register
void gemm_v3(const float *A, const float *B, float *C, int M, int K, int N);

// split 8x8 tile to four 4x4 tile and use 8x4 warp tiling to avoid
// shared memory load bank conflict.
void gemm_v4(const float *A, const float *B, float *C, int M, int K, int N);

// use float4 to access global memory. reduce shared memory store bank conflict
void gemm_v5(const float *A, const float *B, float *C, int M, int K, int N);

// based on v5: double buffer for global load
void gemm_v6(const float *A, const float *B, float *C, int M, int K, int N);

// based on v4: double buffer for global load
void gemm_v6_1(const float *A, const float *B, float *C, int M, int K, int N);

// reduce math stall
// force tile_B shared memory store to use LDS.U.128
void gemm_v7(const float *A, const float *B, float *C, int M, int K, int N);
