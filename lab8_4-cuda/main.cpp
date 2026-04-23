#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <windows.h>

using namespace std;

extern "C" void gpu_mul_naive(const float* A, const float* B, float* C, int N, int S, int UNROLL);
extern "C" void gpu_mul_row_cache(const float* A, const float* B, float* C, int N, int S, int UNROLL);
extern "C" void gpu_mul_col_cache(const float* A, const float* B, float* C, int N, int S, int UNROLL);
extern "C" void gpu_mul_tiled(const float* A, const float* B, float* C, int N, int S, int UNROLL);

void fill_matrix(float* M, int N)
{
    for (int i = 0; i < N * N; i++)
    {
        M[i] = (float)(rand() % 10);
    }
}

void cpu_mul(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

bool compare_matrices(const float* A, const float* B, int N)
{
    const float eps = 1e-3f;

    for (int i = 0; i < N * N; i++)
    {
        if (fabs(A[i] - B[i]) > eps)
            return false;
    }

    return true;
}

double qpc_time_ms_cpu(const float* A, const float* B, float* C, int N)
{
    LARGE_INTEGER freq, t1, t2;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t1);

    cpu_mul(A, B, C, N);

    QueryPerformanceCounter(&t2);

    return 1000.0 * (double)(t2.QuadPart - t1.QuadPart) / (double)freq.QuadPart;
}

double qpc_time_ms_gpu(
    void (*func)(const float*, const float*, float*, int, int, int),
    const float* A, const float* B, float* C, int N, int S, int UNROLL)
{
    LARGE_INTEGER freq, t1, t2;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t1);

    func(A, B, C, N, S, UNROLL);

    QueryPerformanceCounter(&t2);

    return 1000.0 * (double)(t2.QuadPart - t1.QuadPart) / (double)freq.QuadPart;
}

double gflops(double ms, int N)
{
    double sec = ms / 1000.0;
    double ops = 2.0 * (double)N * N * N;
    return ops / sec / 1e9;
}

void print_header()
{
    cout << left
        << setw(8) << "N"
        << setw(8) << "S"
        << setw(10) << "UNROLL"
        << setw(18) << "Algorithm"
        << setw(15) << "Time(ms)"
        << setw(15) << "GFLOPS"
        << setw(12) << "Correct"
        << setw(12) << "Speedup"
        << "\n";

    cout << string(98, '-') << "\n";
}

void print_row(int N, int S, int UNROLL,
    const char* name,
    double time_ms,
    double perf,
    const char* correct,
    double speedup)
{
    cout << left
        << setw(8) << N
        << setw(8) << S
        << setw(10) << UNROLL
        << setw(18) << name
        << setw(15) << time_ms
        << setw(15) << perf
        << setw(12) << correct
        << setw(12) << speedup
        << "\n";
}

void run_one_test(int N, int S, int UNROLL)
{
    if (N % UNROLL != 0)
        return;

    if (S * S > 1024)
        return;

    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C_cpu = new float[N * N];
    float* C_naive = new float[N * N];
    float* C_row = new float[N * N];
    float* C_col = new float[N * N];
    float* C_tiled = new float[N * N];

    fill_matrix(A, N);
    fill_matrix(B, N);

    double t_cpu = qpc_time_ms_cpu(A, B, C_cpu, N);
    double p_cpu = gflops(t_cpu, N);

    double t_naive = qpc_time_ms_gpu(gpu_mul_naive, A, B, C_naive, N, S, UNROLL);
    double p_naive = gflops(t_naive, N);
    bool ok_naive = compare_matrices(C_cpu, C_naive, N);

    double t_row = qpc_time_ms_gpu(gpu_mul_row_cache, A, B, C_row, N, S, UNROLL);
    double p_row = gflops(t_row, N);
    bool ok_row = compare_matrices(C_cpu, C_row, N);

    double t_col = qpc_time_ms_gpu(gpu_mul_col_cache, A, B, C_col, N, S, UNROLL);
    double p_col = gflops(t_col, N);
    bool ok_col = compare_matrices(C_cpu, C_col, N);

    double t_tiled = qpc_time_ms_gpu(gpu_mul_tiled, A, B, C_tiled, N, S, UNROLL);
    double p_tiled = gflops(t_tiled, N);
    bool ok_tiled = compare_matrices(C_cpu, C_tiled, N);

    print_row(N, S, UNROLL, "CPU", t_cpu, p_cpu, "-", 1.0);
    print_row(N, S, UNROLL, "GPU naive", t_naive, p_naive, ok_naive ? "YES" : "NO", t_cpu / t_naive);
    print_row(N, S, UNROLL, "GPU row cache", t_row, p_row, ok_row ? "YES" : "NO", t_cpu / t_row);
    print_row(N, S, UNROLL, "GPU col cache", t_col, p_col, ok_col ? "YES" : "NO", t_cpu / t_col);
    print_row(N, S, UNROLL, "GPU tiled", t_tiled, p_tiled, ok_tiled ? "YES" : "NO", t_cpu / t_tiled);

    cout << string(98, '-') << "\n";

    delete[] A;
    delete[] B;
    delete[] C_cpu;
    delete[] C_naive;
    delete[] C_row;
    delete[] C_col;
    delete[] C_tiled;
}

int main()
{
    srand((unsigned)time(0));

    const int N_values[] = { 256, 512, 1024 };
    const int S_values[] = { 8, 16, 32 };
    const int U_values[] = { 1, 2, 4, 8 };

    const int N_count = sizeof(N_values) / sizeof(N_values[0]);
    const int S_count = sizeof(S_values) / sizeof(S_values[0]);
    const int U_count = sizeof(U_values) / sizeof(U_values[0]);

    print_header();

    for (int i = 0; i < N_count; i++)
    {
        for (int j = 0; j < S_count; j++)
        {
            for (int k = 0; k < U_count; k++)
            {
                run_one_test(N_values[i], S_values[j], U_values[k]);
            }
        }
    }

    system("pause");
    return 0;
}