#include <cuda_runtime.h>
#include <iostream>

#pragma comment(lib, "cudart.lib")

static void printCudaError(const char* message, cudaError_t err) {
    if (err != cudaSuccess) {
        std::cout << message << ": " << cudaGetErrorString(err) << "\n";
    }
}

void printCudaDevicesInfo() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        printCudaError("cudaGetDeviceCount() failed", err);
        return;
    }

    std::cout << "Number of CUDA-capable devices: " << deviceCount << "\n\n";

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found.\n";
        return;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp dp{};
        err = cudaGetDeviceProperties(&dp, i);
        if (err != cudaSuccess) {
            std::cout << "Failed to get properties for device " << i << ": " << cudaGetErrorString(err) << "\n";
            continue;
        }

        int coreClockKHz = 0;
        int memoryClockKHz = 0;

        cudaError_t errClock = cudaDeviceGetAttribute(&coreClockKHz, cudaDevAttrClockRate, i);

        cudaError_t errMemClock = cudaDeviceGetAttribute(&memoryClockKHz, cudaDevAttrMemoryClockRate, i);

        std::cout << "=============================================\n";
        std::cout << "Device #" << i << "\n";
        std::cout << "=============================================\n";

        std::cout << "Name: " << dp.name << "\n";
        std::cout << "Total global memory: " << static_cast<double>(dp.totalGlobalMem) / (1024.0 * 1024.0) << " MB\n";
        std::cout << "Total constant memory: " << static_cast<double>(dp.totalConstMem) / 1024.0 << " KB\n";
        std::cout << "Shared memory per block: " << static_cast<double>(dp.sharedMemPerBlock) / 1024.0 << " KB\n";
        std::cout << "Registers per block: " << dp.regsPerBlock << "\n";
        std::cout << "Warp size: " << dp.warpSize << "\n";
        std::cout << "Maximum threads per block: " << dp.maxThreadsPerBlock << "\n";
        std::cout << "Compute capability: " << dp.major << "." << dp.minor << "\n";
        std::cout << "Number of streaming multiprocessors: " << dp.multiProcessorCount << "\n";

        if (errClock == cudaSuccess) {
            std::cout << "Core clock rate: " << coreClockKHz / 1000.0 << " MHz\n";
        }
        else {
            std::cout << "Core clock rate: not available (" << cudaGetErrorString(errClock) << ")\n";
        }

        if (errMemClock == cudaSuccess) {
            std::cout << "Memory clock rate: " << memoryClockKHz / 1000.0 << " MHz\n";
        }
        else {
            std::cout << "Memory clock rate: not available (" << cudaGetErrorString(errMemClock) << ")\n";
        }

        std::cout << "L2 cache size: " << static_cast<double>(dp.l2CacheSize) / 1024.0 << " KB\n";
        std::cout << "Memory bus width: " << dp.memoryBusWidth << " bits\n";

        std::cout << "Maximum block dimensions:\n";
        std::cout << "  x = " << dp.maxThreadsDim[0] << "\n";
        std::cout << "  y = " << dp.maxThreadsDim[1] << "\n";
        std::cout << "  z = " << dp.maxThreadsDim[2] << "\n";

        std::cout << "Maximum grid dimensions:\n";
        std::cout << "  x = " << dp.maxGridSize[0] << "\n";
        std::cout << "  y = " << dp.maxGridSize[1] << "\n";
        std::cout << "  z = " << dp.maxGridSize[2] << "\n";

        std::cout << "\n";
    }
}