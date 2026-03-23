#include <windows.h>
#include <iostream>
#include <vector>
#include <intrin.h>

const int N = 200;

static void multiplyMatrices(
	std::vector<std::vector<int>>& A,
	std::vector<std::vector<int>>& B,
	std::vector<std::vector<int>>& C
)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			C[i][j] = 0;

			for (int k = 0; k < N; k++)
				C[i][j] += A[i][k] * B[k][j];
		}
}

static void generateMatrix(std::vector<std::vector<int>>& M)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			M[i][j] = rand() % 100;
}

static double calcMin(std::vector<double>& t)
{
	double m = t[0];
	for (double x : t)
		if (x < m) m = x;
	return m;
}

static double calcAvg(std::vector<double>& t)
{
	double s = 0;
	for (double x : t) s += x;
	return s / t.size();
}

int main()
{
	const int K = 1;

	std::vector<double> tick(K), perf(K), tsc(K);

	std::vector<std::vector<int>> A(N, std::vector<int>(N));
	std::vector<std::vector<int>> B(N, std::vector<int>(N));
	std::vector<std::vector<int>> C(N, std::vector<int>(N));

	generateMatrix(A);
	generateMatrix(B);

	// GetTickCount
	for (int i = 0; i < K; i++)
	{
		ULONGLONG start = GetTickCount64();

		multiplyMatrices(A, B, C);

		ULONGLONG end = GetTickCount64();

		tick[i] = end - start;
	}

	// QueryPerformanceCounter
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);

	for (int i = 0; i < K; i++)
	{
		LARGE_INTEGER start, end;

		QueryPerformanceCounter(&start);

		multiplyMatrices(A, B, C);

		QueryPerformanceCounter(&end);

		perf[i] = 1000.0 * (end.QuadPart - start.QuadPart) / freq.QuadPart;
	}

	// RDTSC
	unsigned __int64 start = __rdtsc();
	Sleep(1000);
	unsigned __int64 end = __rdtsc();
	double cpu_freq = (double)(end - start);

	for (int i = 0; i < K; i++)
	{
		unsigned __int64 start = __rdtsc();

		multiplyMatrices(A, B, C);

		unsigned __int64 end = __rdtsc();

		tsc[i] = 1000.0 * (end - start) / cpu_freq;
	}

	std::cout << "\nGetTickCount:\n";
	std::copy(tick.begin(), tick.end(), std::ostream_iterator<double>(std::cout, " "));
	std::cout << std::endl;
	std::cout << "tmin = " << calcMin(tick) << " ms\n";
	std::cout << "tavg = " << calcAvg(tick) << " ms\n";

	std::cout << "\nQueryPerformanceCounter:\n";
	std::copy(perf.begin(), perf.end(), std::ostream_iterator<double>(std::cout, " "));
	std::cout << std::endl;
	std::cout << "tmin = " << calcMin(perf) << " ms\n";
	std::cout << "tavg = " << calcAvg(perf) << " ms\n";

	std::cout << "\nRDTSC:\n";
	std::copy(tsc.begin(), tsc.end(), std::ostream_iterator<double>(std::cout, " "));
	std::cout << std::endl;
	std::cout << "tmin = " << calcMin(tsc) << " ms\n";
	std::cout << "tavg = " << calcAvg(tsc) << " ms\n";

	return 0;
}