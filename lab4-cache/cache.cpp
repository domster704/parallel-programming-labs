#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <intrin.h>

using namespace std;

volatile float g_sink = 0.0f;

enum AccessMode
{
    MODE_SEQ = 0,
    MODE_RANDOM = 1,
    MODE_RANDOM_PRECOMPUTED = 2
};

struct Result
{
    size_t bytes;
    size_t elements;
    double cycles_per_iter;
};

void init_data(float* a, size_t n)
{
    for (size_t i = 0; i < n; i++)
        a[i] = (float)(i % 1000) * 0.1f;
}

void init_index_array(int* idx, size_t n)
{
    for (size_t i = 0; i < n; i++)
        idx[i] = rand() % n;
}

float run_seq(float* a, size_t n)
{
    float sum = 0.0f;

    for (size_t i = 0; i < n; i++)
        sum += a[i];

    g_sink = sum;
    return sum;
}

float run_random(float* a, size_t n)
{
    float sum = 0.0f;

    for (size_t i = 0; i < n; i++)
    {
        int index = rand() % n;
        sum += a[index];
    }

    g_sink = sum;
    return sum;
}

float run_random_precomputed(float* a, int* idx, size_t n)
{
    float sum = 0.0f;

    for (size_t i = 0; i < n; i++)
        sum += a[idx[i]];

    g_sink = sum;
    return sum;
}

Result measure_once(float* a, int* idx, size_t n, AccessMode mode)
{
    unsigned __int64 t1 = __rdtsc();

    if (mode == MODE_SEQ) run_seq(a, n);
    if (mode == MODE_RANDOM) run_random(a, n);
    if (mode == MODE_RANDOM_PRECOMPUTED) run_random_precomputed(a, idx, n);

    unsigned __int64 t2 = __rdtsc();

    Result res;
    res.bytes = n * sizeof(float);
    res.elements = n;
    res.cycles_per_iter = (double)(t2 - t1) / (double)n;

    return res;
}

void append_range(size_t* arr, size_t& count, size_t begin_bytes, size_t end_bytes, size_t step_bytes)
{
    for (size_t b = begin_bytes; b <= end_bytes; b += step_bytes)
        arr[count++] = b;
}

void build_test_sizes(size_t* sizes, size_t& count)
{
    count = 0;

    append_range(sizes, count, 1 * 1024, 2 * 1024 * 1024, 1 * 1024);
    append_range(sizes, count, 2 * 1024 * 1024 + 512 * 1024, 32 * 1024 * 1024, 512 * 1024);
    append_range(sizes, count, 35 * 1024 * 1024, 150 * 1024 * 1024, 5 * 1024 * 1024);
}

const char* mode_name(AccessMode mode)
{
    if (mode == MODE_SEQ) return "sequential";
    if (mode == MODE_RANDOM) return "random";
    if (mode == MODE_RANDOM_PRECOMPUTED) return "random_precomputed";
    return "unknown";
}

const char* mode_file_name(AccessMode mode)
{
    if (mode == MODE_SEQ) return "sequential.csv";
    if (mode == MODE_RANDOM) return "random.csv";
    if (mode == MODE_RANDOM_PRECOMPUTED) return "random_precomputed.csv";
    return "unknown.csv";
}

int main()
{
    srand((unsigned)time(0));

    const size_t MAX_DATA_BYTES = 150ull * 1024ull * 1024ull;
    const size_t MAX_ELEMENTS = MAX_DATA_BYTES / sizeof(float);

    float* data = new float[MAX_ELEMENTS];
    int* index_array = new int[MAX_ELEMENTS];

    init_data(data, MAX_ELEMENTS);
    init_index_array(index_array, MAX_ELEMENTS);

    const size_t MAX_SIZES = 5000;
    size_t* sizes = new size_t[MAX_SIZES];
    size_t sizes_count = 0;
    build_test_sizes(sizes, sizes_count);

    cout << fixed << setprecision(3);

    for (int mode_int = 0; mode_int < 3; mode_int++)
    {
        AccessMode mode = (AccessMode)mode_int;

        ofstream csv(mode_file_name(mode));
        if (!csv)
        {
            cout << "Can not open file " << mode_file_name(mode) << "\n";
            continue;
        }

        csv << "size_bytes,size_kb,size_mb,elements,cycles_per_iteration\n";

        cout << "\n=== MODE: " << mode_name(mode) << " ===\n";

        for (size_t s = 0; s < sizes_count; s++)
        {
            size_t bytes = sizes[s];
            size_t n = bytes / sizeof(float);
            if (n == 0) n = 1;

            Result res = measure_once(data, index_array, n, mode);

            double kb = (double)res.bytes / 1024.0;
            double mb = (double)res.bytes / (1024.0 * 1024.0);

            csv << res.bytes << ","
                << kb << ","
                << mb << ","
                << res.elements << ","
                << res.cycles_per_iter << "\n";

            cout << setw(20) << mode_name(mode)
                << " | size = " << setw(10) << kb << " KB"
                << " | cycles/iter = " << setw(10) << res.cycles_per_iter
                << "\n";
        }

        csv.close();
    }

    delete[] sizes;
    delete[] data;
    delete[] index_array;

    return 0;
}