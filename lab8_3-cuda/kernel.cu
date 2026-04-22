#include <cuda_runtime.h>
#include <windows.h>
#include <iostream>
#include <iomanip>
#include <cstring>

#pragma comment(lib, "cudart.lib")

void runBandwidthTest()
{
    const size_t sizeMB = 64;
    const size_t size = sizeMB * 1024 * 1024;
    const int iterations = 30;

    unsigned char* h1 = (unsigned char*)malloc(size);
    unsigned char* h2 = (unsigned char*)malloc(size);
    unsigned char* h3 = (unsigned char*)malloc(size);

    unsigned char* hp1 = nullptr;
    unsigned char* hp2 = nullptr;

    unsigned char* d1 = nullptr;
    unsigned char* d2 = nullptr;

    if (!h1 || !h2 || !h3)
    {
        std::cout << "Host memory allocation error\n";
        return;
    }

    if (cudaMallocHost((void**)&hp1, size) != cudaSuccess ||
        cudaMallocHost((void**)&hp2, size) != cudaSuccess)
    {
        std::cout << "Pinned memory allocation error\n";
        return;
    }

    if (cudaMalloc((void**)&d1, size) != cudaSuccess ||
        cudaMalloc((void**)&d2, size) != cudaSuccess)
    {
        std::cout << "Device memory allocation error\n";
        return;
    }

    for (size_t i = 0; i < size; i++)
    {
        h1[i] = (unsigned char)(i % 256);
        hp1[i] = (unsigned char)(i % 256);
    }

    LARGE_INTEGER freq, t1, t2;
    QueryPerformanceFrequency(&freq);

    double timeMs, speed;
    bool ok;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Buffer size: " << sizeMB << " MB\n";
    std::cout << "Iterations: " << iterations << "\n\n";

	// RAM -> RAM
    QueryPerformanceCounter(&t1);
    for (int i = 0; i < iterations; i++)
        memcpy(h2, h1, size);
    QueryPerformanceCounter(&t2);

    timeMs = (double)(t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart / iterations;
    speed = ((double)size / 1e9) / (timeMs / 1000.0);
    ok = (memcmp(h1, h2, size) == 0);

    std::cout << "1) RAM -> RAM\n";
    std::cout << "   Correct: " << (ok ? "Yes" : "No") << "\n";
    std::cout << "   Time: " << timeMs << " ms\n";
    std::cout << "   Speed: " << speed << " GB/s\n\n";

    // RAM -> GPU
    QueryPerformanceCounter(&t1);
    for (int i = 0; i < iterations; i++)
        cudaMemcpy(d1, h1, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    QueryPerformanceCounter(&t2);

    cudaMemcpy(h3, d1, size, cudaMemcpyDeviceToHost);
    timeMs = (double)(t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart / iterations;
    speed = ((double)size / 1e9) / (timeMs / 1000.0);
    ok = (memcmp(h1, h3, size) == 0);

    std::cout << "2) RAM -> GPU\n";
    std::cout << "   Correct: " << (ok ? "Yes" : "No") << "\n";
    std::cout << "   Time: " << timeMs << " ms\n";
    std::cout << "   Speed: " << speed << " GB/s\n\n";

    // GPU -> RAM
    cudaMemcpy(d1, h1, size, cudaMemcpyHostToDevice);

    QueryPerformanceCounter(&t1);
    for (int i = 0; i < iterations; i++)
        cudaMemcpy(h2, d1, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    QueryPerformanceCounter(&t2);

    timeMs = (double)(t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart / iterations;
    speed = ((double)size / 1e9) / (timeMs / 1000.0);
    ok = (memcmp(h1, h2, size) == 0);

    std::cout << "3) GPU -> RAM\n";
    std::cout << "   Correct: " << (ok ? "Yes" : "No") << "\n";
    std::cout << "   Time: " << timeMs << " ms\n";
    std::cout << "   Speed: " << speed << " GB/s\n\n";

    // page-locked RAM -> GPU
    QueryPerformanceCounter(&t1);
    for (int i = 0; i < iterations; i++)
        cudaMemcpy(d1, hp1, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    QueryPerformanceCounter(&t2);

    cudaMemcpy(h3, d1, size, cudaMemcpyDeviceToHost);
    timeMs = (double)(t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart / iterations;
    speed = ((double)size / 1e9) / (timeMs / 1000.0);
    ok = (memcmp(hp1, h3, size) == 0);

    std::cout << "4) RAM -> GPU (page-locked)\n";
    std::cout << "   Correct: " << (ok ? "Yes" : "No") << "\n";
    std::cout << "   Time: " << timeMs << " ms\n";
    std::cout << "   Speed: " << speed << " GB/s\n\n";

    // GPU -> page-locked RAM
    cudaMemcpy(d1, hp1, size, cudaMemcpyHostToDevice);

    QueryPerformanceCounter(&t1);
    for (int i = 0; i < iterations; i++)
        cudaMemcpy(hp2, d1, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    QueryPerformanceCounter(&t2);

    timeMs = (double)(t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart / iterations;
    speed = ((double)size / 1e9) / (timeMs / 1000.0);
    ok = (memcmp(hp1, hp2, size) == 0);

    std::cout << "5) GPU -> RAM (page-locked)\n";
    std::cout << "   Correct: " << (ok ? "Yes" : "No") << "\n";
    std::cout << "   Time: " << timeMs << " ms\n";
    std::cout << "   Speed: " << speed << " GB/s\n\n";

    // GPU -> GPU
    cudaMemcpy(d1, h1, size, cudaMemcpyHostToDevice);

    QueryPerformanceCounter(&t1);
    for (int i = 0; i < iterations; i++)
        cudaMemcpy(d2, d1, size, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    QueryPerformanceCounter(&t2);

    cudaMemcpy(h3, d2, size, cudaMemcpyDeviceToHost);
    timeMs = (double)(t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart / iterations;
    speed = ((double)size / 1e9) / (timeMs / 1000.0);
    ok = (memcmp(h1, h3, size) == 0);

    std::cout << "6) GPU -> GPU\n";
    std::cout << "   Correct: " << (ok ? "Yes" : "No") << "\n";
    std::cout << "   Time: " << timeMs << " ms\n";
    std::cout << "   Speed: " << speed << " GB/s\n\n";

    free(h1);
    free(h2);
    free(h3);
    cudaFreeHost(hp1);
    cudaFreeHost(hp2);
    cudaFree(d1);
    cudaFree(d2);
}