#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <windows.h>

using namespace std;

extern "C" void gpu_mul_naive(const float* A, const float* B, float* C, int N,
    int S);
extern "C" void gpu_mul_row_cache(const float* A, const float* B, float* C,
    int N, int S);
extern "C" void gpu_mul_col_cache(const float* A, const float* B, float* C,
    int N, int S);
extern "C" void gpu_mul_tiled(const float* A, const float* B, float* C, int N,
    int S);
extern "C" int gpu_has_error();
extern "C" const char* gpu_get_error_string();

void fill_matrix(float* M, int N) {
    for (int i = 0; i < N * N; i++) {
        M[i] = (float)(rand() % 10);
    }
}

void cpu_mul(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

bool compare_matrices(const float* A, const float* B, int N) {
    const float eps = 1e-3f;

    for (int i = 0; i < N * N; i++) {
        if (fabs(A[i] - B[i]) > eps)
            return false;
    }
    return true;
}

double qpc_time_ms(void (*func)(const float*, const float*, float*, int,
    int),
    const float* A, const float* B, float* C, int N, int S) {
    LARGE_INTEGER freq, t1, t2;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t1);

    func(A, B, C, N, S);

    QueryPerformanceCounter(&t2);

    return 1000.0 * (double)(t2.QuadPart - t1.QuadPart) / (double)freq.QuadPart;
}

double qpc_time_ms_cpu(const float* A, const float* B, float* C, int N) {
    LARGE_INTEGER freq, t1, t2;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t1);

    cpu_mul(A, B, C, N);

    QueryPerformanceCounter(&t2);

    return 1000.0 * (double)(t2.QuadPart - t1.QuadPart) / (double)freq.QuadPart;
}

double gflops(double ms, int N) {
    double sec = ms / 1000.0;
    double ops = 2.0 * N * N * N;
    return ops / sec / 1e9;
}

void run_one_test(int N, int S) {
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

    double t_naive = qpc_time_ms(gpu_mul_naive, A, B, C_naive, N, S);
    double p_naive = gflops(t_naive, N);
    bool ok_naive = compare_matrices(C_cpu, C_naive, N);

    double t_row = qpc_time_ms(gpu_mul_row_cache, A, B, C_row, N, S);
    double p_row = gflops(t_row, N);
    bool ok_row = compare_matrices(C_cpu, C_row, N);

    double t_col = qpc_time_ms(gpu_mul_col_cache, A, B, C_col, N, S);
    double p_col = gflops(t_col, N);
    bool ok_col = compare_matrices(C_cpu, C_col, N);

    bool tiled_done = false;
    double t_tiled = 0.0;
    double p_tiled = 0.0;
    bool ok_tiled = false;

    if (S * S <= 1024) {
        t_tiled = qpc_time_ms(gpu_mul_tiled, A, B, C_tiled, N, S);
        p_tiled = gflops(t_tiled, N);
        ok_tiled = compare_matrices(C_cpu, C_tiled, N);
        tiled_done = true;
    }

    cout << "\nN = " << N << ", S = " << S << "\n";
    cout << left << setw(20) << "Algorithm" << setw(15) << "Time (ms)" << setw(15)
        << "GFLOPS" << setw(12) << "Correct" << setw(12) << "Speedup"
        << "\n";

    cout << left << setw(20) << "CPU" << setw(15) << t_cpu << setw(15) << p_cpu
        << setw(12) << "-" << setw(12) << "1.0"
        << "\n";

    cout << left << setw(20) << "GPU naive" << setw(15) << t_naive << setw(15)
        << p_naive << setw(12) << (ok_naive ? "YES" : "NO") << setw(12)
        << (t_cpu / t_naive) << "\n";

    cout << left << setw(20) << "GPU row cache" << setw(15) << t_row << setw(15)
        << p_row << setw(12) << (ok_row ? "YES" : "NO") << setw(12)
        << (t_cpu / t_row) << "\n";

    cout << left << setw(20) << "GPU col cache" << setw(15) << t_col << setw(15)
        << p_col << setw(12) << (ok_col ? "YES" : "NO") << setw(12)
        << (t_cpu / t_col) << "\n";

    if (tiled_done) {
        cout << left << setw(20) << "GPU tiled" << setw(15) << t_tiled << setw(15)
            << p_tiled << setw(12) << (ok_tiled ? "YES" : "NO") << setw(12)
            << (t_cpu / t_tiled) << "\n";
    }
    else {
        cout << left << setw(20) << "GPU tiled" << setw(15) << "-" << setw(15)
            << "-" << setw(12) << "-" << setw(12) << "-"
            << "\n";
    }

    cout << "\n";

    delete[] A;
    delete[] B;
    delete[] C_cpu;
    delete[] C_naive;
    delete[] C_row;
    delete[] C_col;
    delete[] C_tiled;
}

int main() {
    srand((unsigned)time(0));

    const int N_values[] = { 128, 256, 512, 1024 };
    const int S_values[] = { 4, 8, 16, 32 };

    const int N_count = sizeof(N_values) / sizeof(N_values[0]);
    const int S_count = sizeof(S_values) / sizeof(S_values[0]);

    for (int i = 0; i < N_count; i++) {
        for (int j = 0; j < S_count; j++) {
            run_one_test(N_values[i], S_values[j]);
        }
    }

    system("pause");
    return 0;
}