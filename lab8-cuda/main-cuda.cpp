#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cmath>

extern "C" void vec_add_cuda(const float* a, const float* b, float* c, int n);

using namespace std;

const int N = 1024;

float a[N];
float b[N];
float c_gpu[N];
float c_cpu[N];

void vec_add_cpu(const float* a, const float* b, float* c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

bool compare_vectors(const float* x, const float* y, int n, float eps = 1e-5f)
{
    for (int i = 0; i < n; i++)
    {
        if (fabs(x[i] - y[i]) > eps)
        {
            cout << "Mismatch at index " << i
                << ": CPU = " << x[i]
                << ", GPU = " << y[i] << endl;
            return false;
        }
    }
    return true;
}

int main()
{
    srand(static_cast<unsigned>(time(nullptr)));

    for (int i = 0; i < N; i++)
    {
        a[i] = static_cast<float>(rand() % 1000) / 10.0f;
        b[i] = static_cast<float>(rand() % 1000) / 10.0f;
        c_gpu[i] = 0.0f;
        c_cpu[i] = 0.0f;
    }

    vec_add_cuda(a, b, c_gpu, N);

    vec_add_cpu(a, b, c_cpu, N);

    cout << fixed << setprecision(2);
    cout << "First 20 elements:\n";
    cout << " i        a        b      CPU      GPU\n";
    cout << "---------------------------------------\n";
    for (int i = 0; i < 20; i++)
    {
        cout << setw(2) << i << " "
            << setw(8) << a[i] << " "
            << setw(8) << b[i] << " "
            << setw(8) << c_cpu[i] << " "
            << setw(8) << c_gpu[i] << "\n";
    }

    bool ok = compare_vectors(c_cpu, c_gpu, N);

    cout << "\nResult check: " << (ok ? "OK, CPU and GPU results match."
        : "ERROR, results differ.") << endl;
    
    return 0;
}