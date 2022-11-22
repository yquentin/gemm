static const int BLOCK = 128;
static const int BLOCK_K = 8;
static const int THREAD_TILE = 8;
static const int ele_per_thread = 4;
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

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define C_FETCH_FLOAT4(pointer) \
    (reinterpret_cast<const float4 *>(&(pointer))[0])

__global__ void kernel_v6(int m, int n, int k, const float *__restrict__ a,
        int lda, const float *__restrict__ b, int ldb, float *c, int ldc) {
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

    __shared__ float tile_A[2][BLOCK_K][BLOCK];
    __shared__ float tile_B[2][BLOCK_K][BLOCK];

    float ldg_A[iter_num][4];
    float ldg_B[iter_num][4];

    // pre-load A and B for first iteration
    // load A
    for (int i = 0; i < iter_num; i++) {
        int iter_start_row = i * rows_per_iter_A;
        int _row = iter_start_row + tid / t_pre_row_A;
        int _col = (tid % t_pre_row_A) * ele_per_thread;
        int row = blockIdx.y * BLOCK + _row;
        int col = _col;
        FETCH_FLOAT4(ldg_A[i][0]) = C_FETCH_FLOAT4(*(a + row * lda + col));
        tile_A[0][_col + 0][_row] = ldg_A[i][0];
        tile_A[0][_col + 1][_row] = ldg_A[i][1];
        tile_A[0][_col + 2][_row] = ldg_A[i][2];
        tile_A[0][_col + 3][_row] = ldg_A[i][3];
    }

    // load B
    for (int i = 0; i < iter_num; i++) {
        int iter_start_row = i * rows_per_iter_B;
        int _row = iter_start_row + tid / t_pre_row_B;
        int _col = (tid % t_pre_row_B) * ele_per_thread;
        int row = _row;
        int col = blockIdx.x * BLOCK + _col;
        FETCH_FLOAT4(ldg_B[i][0]) = C_FETCH_FLOAT4(*(b + row * ldb + col));
        tile_B[0][_row][_col] = ldg_B[i][0];
        tile_B[0][_row][_col + 1] = ldg_B[i][1];
        tile_B[0][_row][_col + 2] = ldg_B[i][2];
        tile_B[0][_row][_col + 3] = ldg_B[i][3];
    }

    __syncthreads();

    for (int _k = 0; _k < k - BLOCK_K; _k += BLOCK_K) {
        int write_idx = (_k / BLOCK_K + 1) % 2;
        int read_idx = (_k / BLOCK_K) % 2;

        // load A to register for next iteration
        for (int i = 0; i < iter_num; i++) {
            int iter_start_row = i * rows_per_iter_A;
            int _row = iter_start_row + tid / t_pre_row_A;
            int _col = (tid % t_pre_row_A) * ele_per_thread;
            int row = blockIdx.y * BLOCK + _row;
            int col = _k + BLOCK_K + _col;
            FETCH_FLOAT4(ldg_A[i][0]) = C_FETCH_FLOAT4(*(a + row * lda + col));
        }

        // load B to register for next iteration
        for (int i = 0; i < iter_num; i++) {
            int iter_start_row = i * rows_per_iter_B;
            int _row = iter_start_row + tid / t_pre_row_B;
            int _col = (tid % t_pre_row_B) * ele_per_thread;
            int row = _k + BLOCK_K + _row;
            int col = blockIdx.x * BLOCK + _col;
            FETCH_FLOAT4(ldg_B[i][0]) = C_FETCH_FLOAT4(*(b + row * ldb + col));
        }

#pragma unroll
        for (int s1 = 0; s1 < 2; s1++) {
#pragma unroll
            for (int s2 = 0; s2 < 2; s2++) {
                const int row_start = s1 * 64;
                const int col_start = s2 * 64;
                for (int r = 0; r < BLOCK_K; r++) {
                    for (int i = 0; i < HALF_THREAD_TILE; i++) {
                        int _row = row_start + thread_tile_row_start + i;
                        t_tile_A[i] = tile_A[read_idx][r][_row];
                    }

                    for (int j = 0; j < HALF_THREAD_TILE; j++) {
                        int _col = col_start + thread_tile_col_start + j;
                        t_tile_B[j] = tile_B[read_idx][r][_col];
                    }

                    for (int i = 0; i < HALF_THREAD_TILE; i++) {
                        for (int j = 0; j < HALF_THREAD_TILE; j++) {
                            sum[s1][s2][i][j] += t_tile_A[i] * t_tile_B[j];
                        }
                    }
                }
            }
        }

        // store A to shared mem for next iteration
        for (int i = 0; i < iter_num; i++) {
            int iter_start_row = i * rows_per_iter_A;
            int _row = iter_start_row + tid / t_pre_row_A;
            int _col = (tid % t_pre_row_A) * ele_per_thread;
            int row = blockIdx.y * BLOCK + _row;
            int col = _k + BLOCK_K + _col;
            tile_A[write_idx][_col + 0][_row] = ldg_A[i][0];
            tile_A[write_idx][_col + 1][_row] = ldg_A[i][1];
            tile_A[write_idx][_col + 2][_row] = ldg_A[i][2];
            tile_A[write_idx][_col + 3][_row] = ldg_A[i][3];
        }

        // store B to shared mem for next iteration
        for (int i = 0; i < iter_num; i++) {
            int iter_start_row = i * rows_per_iter_B;
            int _row = iter_start_row + tid / t_pre_row_B;
            int _col = (tid % t_pre_row_B) * ele_per_thread;
            int row = _k + BLOCK_K + _row;
            int col = blockIdx.x * BLOCK + _col;
            tile_B[write_idx][_row][_col] = ldg_B[i][0];
            tile_B[write_idx][_row][_col + 1] = ldg_B[i][1];
            tile_B[write_idx][_row][_col + 2] = ldg_B[i][2];
            tile_B[write_idx][_row][_col + 3] = ldg_B[i][3];
        }

        __syncthreads();
    }

#pragma unroll
    for (int s1 = 0; s1 < 2; s1++) {
#pragma unroll
        for (int s2 = 0; s2 < 2; s2++) {
            const int row_start = s1 * 64;
            const int col_start = s2 * 64;
            for (int r = 0; r < BLOCK_K; r++) {
                for (int i = 0; i < HALF_THREAD_TILE; i++) {
                    int _row = row_start + thread_tile_row_start + i;
                    t_tile_A[i] = tile_A[1][r][_row];
                }

                for (int j = 0; j < HALF_THREAD_TILE; j++) {
                    int _col = col_start + thread_tile_col_start + j;
                    t_tile_B[j] = tile_B[1][r][_col];
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

// store C
#pragma unroll
    for (int s1 = 0; s1 < 2; s1++) {
#pragma unroll
        for (int s2 = 0; s2 < 2; s2++) {
            int c_row_start = BLOCK * blockIdx.y + s1 * 64;
            int c_col_start = BLOCK * blockIdx.x + s2 * 64;
            for (int i = 0; i < HALF_THREAD_TILE; i++) {
                int row = c_row_start + thread_tile_row_start + i;
                for (int j = 0; j < HALF_THREAD_TILE; j += ele_per_thread) {
                    int col = c_col_start + thread_tile_col_start + j;
                    int c_off = row * ldc + col;
                    FETCH_FLOAT4(*(c + c_off))
                            = FETCH_FLOAT4(sum[s1][s2][i][j]);
                }
            }
        }
    }
}

void gemm_v6(const float *A, const float *B, float *C, int M, int K, int N) {
    dim3 dimBlock(block_dim_x, block_dim_y);
    dim3 dimGrid(M / BLOCK, N / BLOCK);
    kernel_v6<<<dimGrid, dimBlock>>>(M, N, K, A, K, B, N, C, N);
}