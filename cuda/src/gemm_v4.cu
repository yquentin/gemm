static const int BLOCK = 128;
static const int BLOCK_K = 8;
static const int THREAD_TILE = 8;
static const int ele_per_thread = 1;
// thread block dims
static const int block_dim_x = BLOCK / THREAD_TILE;
static const int block_dim_y = BLOCK / THREAD_TILE;
// the thread number in a block
static const int thread_num = block_dim_x * block_dim_y;
//
static const int tile_size = BLOCK * BLOCK_K;
// the thread number used to load per row of tile_A
static const int t_pre_row_A = BLOCK_K / ele_per_thread;
// the thread number used to load per row of tile_B
static const int t_pre_row_B = BLOCK / ele_per_thread;
// the iteration number used to load a tile_A or tile_B
static const int iter_num = tile_size / (thread_num * ele_per_thread);
// the rows of tile_A that loaded by all threads in block per iter
static const int rows_per_iter_A = thread_num / t_pre_row_A;
// the rows of tile_B that loaded by all threads in block per iter
static const int rows_per_iter_B = thread_num / t_pre_row_B;

__global__ void kernel_v4(int m, int n, int k, const float *a, int lda,
        const float *b, int ldb, float *c, int ldc) {
    __shared__ float tile_A[BLOCK_K][BLOCK];
    __shared__ float tile_B[BLOCK_K][BLOCK];

    // the linear thread id in a block
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int thread_tile_row_start
            = (warp_id / 4) * 32 + ((lane_id / 2) % 8) * 4;
    const int thread_tile_col_start
            = (warp_id % 4) * 16 + (lane_id / 16) * 8 + (lane_id % 2) * 4;

    const int HALF_THREAD_TILE = THREAD_TILE / 2;

    float sum[2][2][HALF_THREAD_TILE][HALF_THREAD_TILE];
    for (int s1 = 0; s1 < 2; s1++) {
        for (int s2 = 0; s2 < 2; s2++) {
            for (int i = 0; i < HALF_THREAD_TILE; i++) {
                for (int j = 0; j < HALF_THREAD_TILE; j++) {
                    sum[s1][s2][i][j] = 0.f;
                }
            }
        }
    }

    float t_tile_A[HALF_THREAD_TILE];
    float t_tile_B[HALF_THREAD_TILE];

    for (int _k = 0; _k < k; _k += BLOCK_K) {
        // load A
        for (int i = 0; i < iter_num; i++) {
            int iter_start_row = i * rows_per_iter_A;
            int _row = iter_start_row + tid / t_pre_row_A;
            int _col = tid % t_pre_row_A;
            int row = blockIdx.y * BLOCK + _row;
            int col = _k + _col;
            // transposed tile_A
            tile_A[_col][_row] = *(a + row * lda + col);
        }

        // load B
        for (int i = 0; i < iter_num; i++) {
            int iter_start_row = i * rows_per_iter_B;
            int _row = iter_start_row + tid / t_pre_row_B;
            int _col = tid % t_pre_row_B;
            int row = _k + _row;
            int col = blockIdx.x * BLOCK + _col;
            tile_B[_row][_col] = *(b + row * ldb + col);
        }

        __syncthreads(); // wait all threads in a block to finish copy

#pragma unroll
        for (int s1 = 0; s1 < 2; s1++) {
#pragma unroll
            for (int s2 = 0; s2 < 2; s2++) {
                const int row_start = s1 * 64;
                const int col_start = s2 * 64;
                for (int r = 0; r < BLOCK_K; r++) {
                    for (int i = 0; i < HALF_THREAD_TILE; i++) {
                        int _row = row_start + thread_tile_row_start + i;
                        t_tile_A[i] = tile_A[r][_row];
                    }

                    for (int j = 0; j < HALF_THREAD_TILE; j++) {
                        int _col = col_start + thread_tile_col_start + j;
                        t_tile_B[j] = tile_B[r][_col];
                    }

                    for (int i = 0; i < HALF_THREAD_TILE; i++) {
                        for (int j = 0; j < HALF_THREAD_TILE; j++) {
                            sum[s1][s2][i][j] += t_tile_A[i] * t_tile_B[j];
                        }
                    }
                }
            }
        }

        __syncthreads(); // wait all threads to finish compute
    }

// store C
#pragma unroll
    for (int s1 = 0; s1 < 2; s1++) {
#pragma unroll
        for (int s2 = 0; s2 < 2; s2++) {
            int c_row_start = BLOCK * blockIdx.y + s1 * 64;
            int c_col_start = BLOCK * blockIdx.x + s2 * 64;
            for (int i = 0; i < HALF_THREAD_TILE; i++) {
                int row = c_row_start + thread_tile_row_start + i;
                for (int j = 0; j < HALF_THREAD_TILE; j++) {
                    int col = c_col_start + thread_tile_col_start + j;
                    int c_off = row * ldc + col;
                    *(c + c_off) = sum[s1][s2][i][j];
                }
            }
        }
    }
}

void gemm_v4(const float *A, const float *B, float *C, int M, int K, int N) {
    dim3 dimBlock(block_dim_x, block_dim_y);
    dim3 dimGrid(M / BLOCK, N / BLOCK);
    kernel_v4<<<dimGrid, dimBlock>>>(M, N, K, A, K, B, N, C, N);
}