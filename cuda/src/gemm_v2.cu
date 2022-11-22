static const int BLOCK = 128;
static const int BLOCK_K = 8;
static const int THREAD_TILE = 4;
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

__global__ void kernel_v2(int m, int n, int k, const float *a, int lda,
        const float *b, int ldb, float *c, int ldc) {
    __shared__ float tile_A[BLOCK][BLOCK_K]; // 128x8
    __shared__ float tile_B[BLOCK_K][BLOCK]; // 8x128

    // the linear thread id in a block
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    float sum[THREAD_TILE][THREAD_TILE];
    for (int i = 0; i < THREAD_TILE; i++) {
        for (int j = 0; j < THREAD_TILE; j++) {
            sum[i][j] = 0.f;
        }
    }

    for (int _k = 0; _k < k; _k += BLOCK_K) {
        // load A
        for (int i = 0; i < iter_num; i++) {
            int iter_start_row = i * rows_per_iter_A;
            int _row = iter_start_row + tid / t_pre_row_A;
            int _col = tid % t_pre_row_A;
            int row = blockIdx.y * BLOCK + _row;
            int col = _k + _col;
            tile_A[_row][_col] = *(a + row * lda + col);
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

        // compute gemm on tile (each thread compute [THREAD_TILE, THREAD_TILE] block)
        int thread_tile_row_start = threadIdx.y * THREAD_TILE;
        int thread_tile_col_start = threadIdx.x * THREAD_TILE;
        for (int i = 0; i < THREAD_TILE; i++) {
            int _row = thread_tile_row_start + i;
            for (int j = 0; j < THREAD_TILE; j++) {
                int _col = thread_tile_col_start + j;
                for (int r = 0; r < BLOCK_K; r++) {
                    sum[i][j] += tile_A[_row][r] * tile_B[r][_col];
                }
            }
        }

        __syncthreads(); // wait all threads to finish compute
    }

    // store C
    int c_row_start = BLOCK * blockIdx.y;
    int c_col_start = BLOCK * blockIdx.x;
    for (int i = 0; i < THREAD_TILE; i++) {
        int row = c_row_start + threadIdx.y * THREAD_TILE + i;
        for (int j = 0; j < THREAD_TILE; j++) {
            int col = c_col_start + threadIdx.x * THREAD_TILE + j;
            int c_off = row * ldc + col;
            *(c + c_off) = sum[i][j];
        }
    }
}

void gemm_v2(const float *A, const float *B, float *C, int M, int K, int N) {
    dim3 dimBlock(block_dim_x, block_dim_y);
    dim3 dimGrid(M / BLOCK, N / BLOCK);
    kernel_v2<<<dimGrid, dimBlock>>>(M, N, K, A, K, B, N, C, N);
}