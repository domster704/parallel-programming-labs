#include <cuda.h>
#include <cuda_runtime.h>

__device__ float dot_naive_unroll(const float* A, const float* B, int row,
    int col, int N, int UNROLL) {
    float sum = 0.0f;
    int k = 0;

    if (UNROLL == 8) {
        for (; k + 7 < N; k += 8) {
            sum += A[row * N + k] * B[(k + 0) * N + col];
            sum += A[row * N + k + 1] * B[(k + 1) * N + col];
            sum += A[row * N + k + 2] * B[(k + 2) * N + col];
            sum += A[row * N + k + 3] * B[(k + 3) * N + col];
            sum += A[row * N + k + 4] * B[(k + 4) * N + col];
            sum += A[row * N + k + 5] * B[(k + 5) * N + col];
            sum += A[row * N + k + 6] * B[(k + 6) * N + col];
            sum += A[row * N + k + 7] * B[(k + 7) * N + col];
        }
    }
    else if (UNROLL == 4) {
        for (; k + 3 < N; k += 4) {
            sum += A[row * N + k] * B[(k + 0) * N + col];
            sum += A[row * N + k + 1] * B[(k + 1) * N + col];
            sum += A[row * N + k + 2] * B[(k + 2) * N + col];
            sum += A[row * N + k + 3] * B[(k + 3) * N + col];
        }
    }
    else if (UNROLL == 2) {
        for (; k + 1 < N; k += 2) {
            sum += A[row * N + k] * B[(k + 0) * N + col];
            sum += A[row * N + k + 1] * B[(k + 1) * N + col];
        }
    }
    else {
        for (; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        return sum;
    }

    for (; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }

    return sum;
}

__device__ float dot_row_cache_unroll(const float* rowPart, const float* B,
    int col, int N, int m, int limit,
    int UNROLL) {
    float sum = 0.0f;
    int k = 0;

    if (UNROLL == 8) {
        for (; k + 7 < limit; k += 8) {
            sum += rowPart[k + 0] * B[(m + k + 0) * N + col];
            sum += rowPart[k + 1] * B[(m + k + 1) * N + col];
            sum += rowPart[k + 2] * B[(m + k + 2) * N + col];
            sum += rowPart[k + 3] * B[(m + k + 3) * N + col];
            sum += rowPart[k + 4] * B[(m + k + 4) * N + col];
            sum += rowPart[k + 5] * B[(m + k + 5) * N + col];
            sum += rowPart[k + 6] * B[(m + k + 6) * N + col];
            sum += rowPart[k + 7] * B[(m + k + 7) * N + col];
        }
    }
    else if (UNROLL == 4) {
        for (; k + 3 < limit; k += 4) {
            sum += rowPart[k + 0] * B[(m + k + 0) * N + col];
            sum += rowPart[k + 1] * B[(m + k + 1) * N + col];
            sum += rowPart[k + 2] * B[(m + k + 2) * N + col];
            sum += rowPart[k + 3] * B[(m + k + 3) * N + col];
        }
    }
    else if (UNROLL == 2) {
        for (; k + 1 < limit; k += 2) {
            sum += rowPart[k + 0] * B[(m + k + 0) * N + col];
            sum += rowPart[k + 1] * B[(m + k + 1) * N + col];
        }
    }
    else {
        for (; k < limit; k++) {
            sum += rowPart[k] * B[(m + k) * N + col];
        }
        return sum;
    }

    for (; k < limit; k++) {
        sum += rowPart[k] * B[(m + k) * N + col];
    }

    return sum;
}

__device__ float dot_col_cache_unroll(const float* A, const float* colPart,
    int row, int N, int m, int limit,
    int UNROLL) {
    float sum = 0.0f;
    int k = 0;

    if (UNROLL == 8) {
        for (; k + 7 < limit; k += 8) {
            sum += A[row * N + (m + k + 0)] * colPart[k + 0];
            sum += A[row * N + (m + k + 1)] * colPart[k + 1];
            sum += A[row * N + (m + k + 2)] * colPart[k + 2];
            sum += A[row * N + (m + k + 3)] * colPart[k + 3];
            sum += A[row * N + (m + k + 4)] * colPart[k + 4];
            sum += A[row * N + (m + k + 5)] * colPart[k + 5];
            sum += A[row * N + (m + k + 6)] * colPart[k + 6];
            sum += A[row * N + (m + k + 7)] * colPart[k + 7];
        }
    }
    else if (UNROLL == 4) {
        for (; k + 3 < limit; k += 4) {
            sum += A[row * N + (m + k + 0)] * colPart[k + 0];
            sum += A[row * N + (m + k + 1)] * colPart[k + 1];
            sum += A[row * N + (m + k + 2)] * colPart[k + 2];
            sum += A[row * N + (m + k + 3)] * colPart[k + 3];
        }
    }
    else if (UNROLL == 2) {
        for (; k + 1 < limit; k += 2) {
            sum += A[row * N + (m + k + 0)] * colPart[k + 0];
            sum += A[row * N + (m + k + 1)] * colPart[k + 1];
        }
    }
    else {
        for (; k < limit; k++) {
            sum += A[row * N + (m + k)] * colPart[k];
        }
        return sum;
    }

    for (; k < limit; k++) {
        sum += A[row * N + (m + k)] * colPart[k];
    }

    return sum;
}

__device__ float dot_tiled_unroll(const float* As, const float* Bs, int ty,
    int tx, int S, int UNROLL) {
    float sum = 0.0f;
    int k = 0;

    if (UNROLL == 8) {
        for (; k + 7 < S; k += 8) {
            sum += As[ty * S + k + 0] * Bs[(k + 0) * S + tx];
            sum += As[ty * S + k + 1] * Bs[(k + 1) * S + tx];
            sum += As[ty * S + k + 2] * Bs[(k + 2) * S + tx];
            sum += As[ty * S + k + 3] * Bs[(k + 3) * S + tx];
            sum += As[ty * S + k + 4] * Bs[(k + 4) * S + tx];
            sum += As[ty * S + k + 5] * Bs[(k + 5) * S + tx];
            sum += As[ty * S + k + 6] * Bs[(k + 6) * S + tx];
            sum += As[ty * S + k + 7] * Bs[(k + 7) * S + tx];
        }
    }
    else if (UNROLL == 4) {
        for (; k + 3 < S; k += 4) {
            sum += As[ty * S + k + 0] * Bs[(k + 0) * S + tx];
            sum += As[ty * S + k + 1] * Bs[(k + 1) * S + tx];
            sum += As[ty * S + k + 2] * Bs[(k + 2) * S + tx];
            sum += As[ty * S + k + 3] * Bs[(k + 3) * S + tx];
        }
    }
    else if (UNROLL == 2) {
        for (; k + 1 < S; k += 2) {
            sum += As[ty * S + k + 0] * Bs[(k + 0) * S + tx];
            sum += As[ty * S + k + 1] * Bs[(k + 1) * S + tx];
        }
    }
    else {
        for (; k < S; k++) {
            sum += As[ty * S + k] * Bs[k * S + tx];
        }
        return sum;
    }

    for (; k < S; k++) {
        sum += As[ty * S + k] * Bs[k * S + tx];
    }

    return sum;
}

__global__ void MatMulNaiveKernel(const float* A, const float* B, float* C,
    int N, int UNROLL) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        C[row * N + col] = dot_naive_unroll(A, B, row, col, N, UNROLL);
    }
}

__global__ void MatMulRowCacheKernel(const float* A, const float* B, float* C,
    int N, int UNROLL) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ float rowPart[];

    float sum = 0.0f;

    for (int m = 0; m < N; m += blockDim.x) {
        int kLoad = m + threadIdx.x;

        if (row < N && kLoad < N)
            rowPart[threadIdx.x] = A[row * N + kLoad];
        else
            rowPart[threadIdx.x] = 0.0f;

        __syncthreads();

        int limit = blockDim.x;
        if (m + limit > N)
            limit = N - m;

        if (row < N && col < N) {
            sum += dot_row_cache_unroll(rowPart, B, col, N, m, limit, UNROLL);
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void MatMulColCacheKernel(const float* A, const float* B, float* C,
    int N, int UNROLL) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;

    extern __shared__ float colPart[];

    float sum = 0.0f;

    for (int m = 0; m < N; m += blockDim.x) {
        int kLoad = m + threadIdx.x;

        if (col < N && kLoad < N)
            colPart[threadIdx.x] = B[kLoad * N + col];
        else
            colPart[threadIdx.x] = 0.0f;

        __syncthreads();

        int limit = blockDim.x;
        if (m + limit > N)
            limit = N - m;

        if (row < N && col < N) {
            sum += dot_col_cache_unroll(A, colPart, row, N, m, limit, UNROLL);
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void MatMulTiledKernel(const float* A, const float* B, float* C,
    int N, int S, int UNROLL) {
    extern __shared__ float shared[];

    float* As = shared;
    float* Bs = shared + S * S;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * S + tx;
    int row = blockIdx.y * S + ty;

    float sum = 0.0f;

    for (int m = 0; m < N; m += S) {
        if (row < N && (m + tx) < N)
            As[ty * S + tx] = A[row * N + (m + tx)];
        else
            As[ty * S + tx] = 0.0f;

        if (col < N && (m + ty) < N)
            Bs[ty * S + tx] = B[(m + ty) * N + col];
        else
            Bs[ty * S + tx] = 0.0f;

        __syncthreads();

        sum += dot_tiled_unroll(As, Bs, ty, tx, S, UNROLL);

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" void gpu_mul_naive(const float* A, const float* B, float* C, int N,
    int S, int UNROLL) {
    size_t bytes = (size_t)N * N * sizeof(float);

    float* dA = 0;
    float* dB = 0;
    float* dC = 0;

    cudaMalloc((void**)&dA, bytes);
    cudaMalloc((void**)&dB, bytes);
    cudaMalloc((void**)&dC, bytes);

    cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);

    dim3 threads(S, S);
    dim3 blocks((N + S - 1) / S, (N + S - 1) / S);

    MatMulNaiveKernel << <blocks, threads >> > (dA, dB, dC, N, UNROLL);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

extern "C" void gpu_mul_row_cache(const float* A, const float* B, float* C,
    int N, int S, int UNROLL) {
    size_t bytes = (size_t)N * N * sizeof(float);

    float* dA = 0;
    float* dB = 0;
    float* dC = 0;

    cudaMalloc((void**)&dA, bytes);
    cudaMalloc((void**)&dB, bytes);
    cudaMalloc((void**)&dC, bytes);

    cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);

    dim3 threads(S);
    dim3 blocks(N, (N + S - 1) / S);

    MatMulRowCacheKernel << <blocks, threads, S * sizeof(float) >> > (dA, dB, dC, N,
        UNROLL);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

extern "C" void gpu_mul_col_cache(const float* A, const float* B, float* C,
    int N, int S, int UNROLL) {
    size_t bytes = (size_t)N * N * sizeof(float);

    float* dA = 0;
    float* dB = 0;
    float* dC = 0;

    cudaMalloc((void**)&dA, bytes);
    cudaMalloc((void**)&dB, bytes);
    cudaMalloc((void**)&dC, bytes);

    cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);

    dim3 threads(S);
    dim3 blocks((N + S - 1) / S, N);

    MatMulColCacheKernel << <blocks, threads, S * sizeof(float) >> > (dA, dB, dC, N,
        UNROLL);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

extern "C" void gpu_mul_tiled(const float* A, const float* B, float* C, int N,
    int S, int UNROLL) {
    size_t bytes = (size_t)N * N * sizeof(float);

    float* dA = 0;
    float* dB = 0;
    float* dC = 0;

    cudaMalloc((void**)&dA, bytes);
    cudaMalloc((void**)&dB, bytes);
    cudaMalloc((void**)&dC, bytes);

    cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);

    dim3 threads(S, S);
    dim3 blocks((N + S - 1) / S, (N + S - 1) / S);

    size_t sharedBytes = 2 * S * S * sizeof(float);

    MatMulTiledKernel << <blocks, threads, sharedBytes >> > (dA, dB, dC, N, S, UNROLL);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}