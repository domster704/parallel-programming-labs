#include <windows.h>
#include <cstdio>
#include "matrix_mul.hpp"


static void printLine(const char* name, int N, double ms)
{
	printf("%-22s N=%-5d time=%10.3f ms  perf=%10.3f GFLOP/s\n",
		name, N, ms, calcGFLOPS(N, ms));
}

int main()
{
	int N = 4096;
	int unrollValues[5] = { 1, 2, 4, 8, 16 };

	double* A = allocateMatrix(N);
	double* B = allocateMatrix(N);
	double* BT = allocateMatrix(N);
	double* C1 = allocateMatrix(N);
	double* C2 = allocateMatrix(N);
	double* C3 = allocateMatrix(N);
	double* C4 = allocateMatrix(N);

	if (!A || !B || !BT || !C1 || !C2 || !C3 || !C4)
	{
		printf("Memory allocation error.\n");
		freeMatrix(A); freeMatrix(B); freeMatrix(BT);
		freeMatrix(C1); freeMatrix(C2); freeMatrix(C3); freeMatrix(C4);
		return 1;
	}
	printf("Fill matrix\n");
	fillMatrix(A, N);
	fillMatrix(B, N);
	

	printf("Calculate matrix\n");
	// Classic
	double tClassic = measure(multiplyClassic, A, B, C1, N, 0);

	// Transposed
	LARGE_INTEGER freq, start, end;
	QueryPerformanceFrequency(&freq);

	QueryPerformanceCounter(&start);
	transposeMatrix(B, BT, N);
	multiplyTransposed(A, BT, C2, N, 0);
	QueryPerformanceCounter(&end);
	double tTransWith = 1000.0 * (double)(end.QuadPart - start.QuadPart) / (double)freq.QuadPart;

	double tTransOnly = measure(multiplyTransposed, A, BT, C2, N, 0);

	// Buffered: choose best M
	double bestBufferedTime = 0.0;
	int bestBufferedM = 1;

	for (int i = 0; i < 5; i++)
	{
		int M = unrollValues[i];
		double t = measure(multiplyBuffered, A, B, C3, N, M);
		printf("Buffered: M=%d, N=%d, t=%.3f ms, P=%.3f GFLOP/s\n",
			M, N, t, calcGFLOPS(N, t));
		if (i == 0 || t < bestBufferedTime)
		{
			bestBufferedTime = t;
			bestBufferedM = M;
		}
	}
	double tBuffered = measure(multiplyBuffered, A, B, C3, N, bestBufferedM);

	// Blocked: choose best S and M
	double bestBlockTime = 0.0;
	int bestS = 1, bestM = 1;
	bool first = true;

	for (int S = 1; S <= N; S *= 2)
	{
		for (int i = 0; i < 5; i++)
		{
			blockUnrollM = unrollValues[i];
			double t = measure(multiplyBlocked, A, B, C4, N, S);
			printf("Blocked: S=%d, M=%d, N=%d, t=%.3f ms, P=%.3f GFLOP/s\n",
				S, blockUnrollM, N, t, calcGFLOPS(N, t));

			if (first || t < bestBlockTime)
			{
				first = false;
				bestBlockTime = t;
				bestS = S;
				bestM = unrollValues[i];
			}
		}
		if (S > N / 2) break;
	}

	blockUnrollM = bestM;
	double tBlocked = measure(multiplyBlocked, A, B, C4, N, bestS);

	bool ok2 = compareMatrices(C1, C2, N, 1e-6);
	bool ok3 = compareMatrices(C1, C3, N, 1e-6);
	bool ok4 = compareMatrices(C1, C4, N, 1e-6);

	printf("Matrix multiplication benchmark\n");
	printf("N = %d\n\n", N);

	printLine("Classic", N, tClassic);
	printLine("Transposed (+prep)", N, tTransWith);
	printLine("Transposed only", N, tTransOnly);
	printLine("Buffered", N, tBuffered);
	printLine("Blocked", N, tBlocked);

	printf("\nBest parameters:\n");
	printf("Buffered: M = %d\n", bestBufferedM);
	printf("Blocked : S = %d, M = %d\n", bestS, bestM);

	printf("\nCorrectness:\n");
	printf("Classic vs Transposed : %s\n", ok2 ? "OK" : "FAIL");
	printf("Classic vs Buffered   : %s\n", ok3 ? "OK" : "FAIL");
	printf("Classic vs Blocked    : %s\n", ok4 ? "OK" : "FAIL");

	freeMatrix(A);
	freeMatrix(B);
	freeMatrix(BT);
	freeMatrix(C1);
	freeMatrix(C2);
	freeMatrix(C3); 
	freeMatrix(C4);

	return 0;
}