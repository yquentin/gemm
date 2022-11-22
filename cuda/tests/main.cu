#include <algorithm>
#include <assert.h>
#include <functional>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "gemm.hpp"

#define NITER 10

#define checkCudaErrors(func) \
    { \
        cudaError_t e = (func); \
        if (e != cudaSuccess) \
            printf("%s %d CUDA: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(e)); \
    }

// Use cuBlas to compute reference results
// A, B and C should be host pointer
void host_compute_ref(
        const float *A, const float *B, float *C, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            C[m * N + n] = 0.f;
        }
    }

    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) {
                C[m * N + n] += A[m * K + k] * B[k * N + n];
            }
        }
    }
}

// cuBlas sgemm bench
// A, B and C should be device pointer
double cublas_gemm(
        const float *A, const float *B, float *C, int M, int K, int N) {
    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    float msecTotal = 0;

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0; run < NITER; run++) {
        cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                N, A, K, &beta, C, N);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    cublasDestroy(blas_handle);

    return msecTotal / NITER;
}

typedef void (*gemm_func)(const float *, const float *, float *, int, int, int);
// My gemm bench
// A, B and C should be device pointer
double my_gemm(const char *alg, const float *A, const float *B, float *C, int M,
        int K, int N) {
    gemm_func func = nullptr;
    if (strcmp(alg, "v0") == 0) {
        func = gemm_v0;
    } else if (strcmp(alg, "v1") == 0) {
        func = gemm_v1;
    } else if (strcmp(alg, "v2") == 0) {
        func = gemm_v2;
    } else if (strcmp(alg, "v3") == 0) {
        func = gemm_v3;
    } else if (strcmp(alg, "v4") == 0) {
        func = gemm_v4;
    } else if (strcmp(alg, "v5") == 0) {
        func = gemm_v5;
    } else if (strcmp(alg, "v6") == 0) {
        func = gemm_v6;
    } else if (strcmp(alg, "v7") == 0) {
        func = gemm_v7;
    } else {
        // TODO
    }

    float msecTotal = 0;

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0; run < NITER; run++) {
        func(A, B, C, M, K, N);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    checkCudaErrors(cudaGetLastError());

    return msecTotal / NITER;
}

// usage: ./main [alg] [M] [K] [N]
int main(int argc, char **argv) {
    const char *alg = argv[1];

    int M = 1024, K = 1024, N = 1024;
    if (argc == 5) {
        M = atoi(argv[2]);
        K = atoi(argv[3]);
        N = atoi(argv[4]);
    }
    double flopsPerMatrixMul = 2.0 * M * N * K;

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;

    std::vector<float> h_A_data(bytes_A / sizeof(float));
    std::vector<float> h_B_data(bytes_B / sizeof(float));
    std::vector<float> h_C_data(bytes_C / sizeof(float));
    std::vector<float> h_C_ref_data(bytes_C / sizeof(float));

    float *h_A = h_A_data.data();
    float *h_B = h_B_data.data();
    float *h_C = h_C_data.data();
    float *h_C_ref = h_C_ref_data.data();

    // generate A and B
    std::generate(h_A_data.begin(), h_A_data.end(), [&]() {
        static int i = 0;
        return (i++) / M + 1;
    });

    std::generate(h_B_data.begin(), h_B_data.end(), [&]() {
        static int i = 0;
        return (i++) % N;
    });

    // compute reference on host
    host_compute_ref(h_A, h_B, h_C_ref, M, K, N);

    // compute on device
    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));
    checkCudaErrors(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    double msecPerMatrixMul;
    if (strcmp(alg, "cublas") == 0) {
        msecPerMatrixMul = cublas_gemm(d_A, d_B, d_C, M, K, N);
    } else {
        msecPerMatrixMul = my_gemm(alg, d_A, d_B, d_C, M, K, N);
    }

    // copy results back to host from device
    checkCudaErrors(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    double gigaFlops
            = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("%s, Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f "
           "Ops\n",
            alg, gigaFlops, msecPerMatrixMul, flopsPerMatrixMul);

    double eps = 1.e-6; // machine zero
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        double abs_err = fabs(h_C_data[i] - h_C_ref_data[i]);
        double dot_length = M;
        double abs_val = fabs(h_C_data[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i,
                    h_C_data[i], h_C_ref_data[i], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}