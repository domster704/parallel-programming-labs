#include <cuda.h>
#include <cuda_runtime.h>

__global__ void MatMulNaiveKernel(const float* A, const float* B, float* C,
    int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void MatMulRowCacheKernel(const float* A, const float* B, float* C,
    int N) {
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
            for (int k = 0; k < limit; k++) {
                sum += rowPart[k] * B[(m + k) * N + col];
            }
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void MatMulColCacheKernel(const float* A, const float* B, float* C,
    int N) {
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
            for (int k = 0; k < limit; k++) {
                sum += A[row * N + (m + k)] * colPart[k];
            }
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void MatMulTiledKernel(const float* A, const float* B, float* C,
    int N, int S) {
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

        for (int k = 0; k < S; k++) {
            sum += As[ty * S + k] * Bs[k * S + tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" void gpu_mul_naive(const float* A, const float* B, float* C, int N,
    int S) {
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

    MatMulNaiveKernel << <blocks, threads >> > (dA, dB, dC, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

extern "C" void gpu_mul_row_cache(const float* A, const float* B, float* C,
    int N, int S) {
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

    MatMulRowCacheKernel << <blocks, threads, S * sizeof(float) >> > (dA, dB, dC, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

extern "C" void gpu_mul_col_cache(const float* A, const float* B, float* C,
    int N, int S) {
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

    MatMulColCacheKernel << <blocks, threads, S * sizeof(float) >> > (dA, dB, dC, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

extern "C" void gpu_mul_tiled(const float* A, const float* B, float* C, int N,
    int S) {
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

    MatMulTiledKernel << <blocks, threads, sharedBytes >> > (dA, dB, dC, N, S);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

extern "C" int gpu_has_error() {
    cudaError_t err = cudaGetLastError();
    return (err != cudaSuccess) ? 1 : 0;
}

extern "C" const char* gpu_get_error_string() {
    cudaError_t err = cudaGetLastError();
    return cudaGetErrorString(err);
}