#include <new>
#include <windows.h>
#define IDX(i, j, N) ((i) * (N) + (j))


double* allocateMatrix(int N)
{
	return new(std::nothrow) double[(long long)N * N];
}

void freeMatrix(double* M)
{
	delete[] M;
}

void zeroMatrix(double* M, int N)
{
	long long size = (long long)N * N;
	for (long long i = 0; i < size; i++) M[i] = 0.0;
}

void fillMatrix(double* M, int N)
{
	double lut[100];
	for (int x = 0; x < 100; ++x)
		lut[x] = x / 10.0 + 1.0;

	for (int i = 0; i < N; ++i)
	{
		double* row = M + (long long)i * N;
		int v = (i * 13) % 100;

		for (int j = 0; j < N; ++j)
		{
			row[j] = lut[v];
			v += 7;
			if (v >= 100) v -= 100;
		}
	}
}

bool compareMatrices(const double* A, const double* B, int N, double eps)
{
	long long size = (long long)N * N;
	for (long long i = 0; i < size; i++)
		if (fabs(A[i] - B[i]) > eps) return false;
	return true;
}

void transposeMatrix(const double* B, double* BT, int N)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			BT[IDX(j, i, N)] = B[IDX(i, j, N)];
}

double calcGFLOPS(int N, double ms)
{
	double sec = ms / 1000.0;
	if (sec <= 0.0) return 0.0;
	return (2.0 * N * N * N) / sec / 1e9;
}

double measure(void (*func)(const double*, const double*, double*, int, int),
	const double* A, const double* B, double* C, int N, int param)
{
	LARGE_INTEGER freq, start, end;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);

	func(A, B, C, N, param);

	QueryPerformanceCounter(&end);
	return 1000.0 * (double)(end.QuadPart - start.QuadPart) / (double)freq.QuadPart;
}


double dotProductUnrolled(const double* X, const double* Y, int len, int M)
{
	double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
	double s5 = 0.0, s6 = 0.0, s7 = 0.0, s8 = 0.0;
	double s9 = 0.0, s10 = 0.0, s11 = 0.0, s12 = 0.0;
	double s13 = 0.0, s14 = 0.0, s15 = 0.0, s16 = 0.0;

	int k = 0;

	if (M <= 1)
	{
		for (; k < len; k++) s1 += X[k] * Y[k];
	}
	else if (M == 2)
	{
		for (; k + 1 < len; k += 2)
		{
			s1 += X[k] * Y[k];
			s2 += X[k + 1] * Y[k + 1];
		}
		for (; k < len; k++) s1 += X[k] * Y[k];
	}
	else if (M == 4)
	{
		for (; k + 3 < len; k += 4)
		{
			s1 += X[k] * Y[k];
			s2 += X[k + 1] * Y[k + 1];
			s3 += X[k + 2] * Y[k + 2];
			s4 += X[k + 3] * Y[k + 3];
		}
		for (; k < len; k++) s1 += X[k] * Y[k];
	}
	else if (M == 8)
	{
		for (; k + 7 < len; k += 8)
		{
			s1 += X[k] * Y[k];
			s2 += X[k + 1] * Y[k + 1];
			s3 += X[k + 2] * Y[k + 2];
			s4 += X[k + 3] * Y[k + 3];
			s5 += X[k + 4] * Y[k + 4];
			s6 += X[k + 5] * Y[k + 5];
			s7 += X[k + 6] * Y[k + 6];
			s8 += X[k + 7] * Y[k + 7];
		}
		for (; k < len; k++) s1 += X[k] * Y[k];
	}
	else
	{
		for (; k + 15 < len; k += 16)
		{
			s1 += X[k] * Y[k];
			s2 += X[k + 1] * Y[k + 1];
			s3 += X[k + 2] * Y[k + 2];
			s4 += X[k + 3] * Y[k + 3];
			s5 += X[k + 4] * Y[k + 4];
			s6 += X[k + 5] * Y[k + 5];
			s7 += X[k + 6] * Y[k + 6];
			s8 += X[k + 7] * Y[k + 7];
			s9 += X[k + 8] * Y[k + 8];
			s10 += X[k + 9] * Y[k + 9];
			s11 += X[k + 10] * Y[k + 10];
			s12 += X[k + 11] * Y[k + 11];
			s13 += X[k + 12] * Y[k + 12];
			s14 += X[k + 13] * Y[k + 13];
			s15 += X[k + 14] * Y[k + 14];
			s16 += X[k + 15] * Y[k + 15];
		}
		for (; k < len; k++) s1 += X[k] * Y[k];
	}

	return s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 +
		s9 + s10 + s11 + s12 + s13 + s14 + s15 + s16;
}

// Classic
void multiplyClassic(const double* A, const double* B, double* C, int N, int unused)
{
	(void)unused;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			double s = 0.0;
			for (int k = 0; k < N; k++)
				s += A[IDX(i, k, N)] * B[IDX(k, j, N)];
			C[IDX(i, j, N)] = s;
		}
	}
}

// Transposed B
void multiplyTransposed(const double* A, const double* BT, double* C, int N, int unused)
{
	(void)unused;
	for (int i = 0; i < N; i++)
	{
		const double* rowA = &A[IDX(i, 0, N)];
		for (int j = 0; j < N; j++)
		{
			const double* rowBT = &BT[IDX(j, 0, N)];
			double s = 0.0;
			for (int k = 0; k < N; k++)
				s += rowA[k] * rowBT[k];
			C[IDX(i, j, N)] = s;
		}
	}
}

// Column buffer
void multiplyBuffered(const double* A, const double* B, double* C, int N, int M)
{
	double* tmp = new(std::nothrow) double[N];
	if (!tmp) return;

	for (int j = 0; j < N; j++)
	{
		for (int k = 0; k < N; k++) tmp[k] = B[IDX(k, j, N)];

		for (int i = 0; i < N; i++)
		{
			const double* rowA = &A[IDX(i, 0, N)];
			C[IDX(i, j, N)] = dotProductUnrolled(rowA, tmp, N, M);
		}
	}

	delete[] tmp;
}

int blockUnrollM = 1;

// Blocked
void multiplyBlocked(const double* A, const double* B, double* C, int N, int S)
{
	if (S < 1) S = 1;
	zeroMatrix(C, N);

	double* blockA = new(std::nothrow) double[S * S];
	double* blockBT = new(std::nothrow) double[S * S];
	if (!blockA || !blockBT)
	{
		delete[] blockA;
		delete[] blockBT;
		return;
	}

	for (int bi = 0; bi < N; bi += S)
	{
		int iSize = (bi + S <= N) ? S : (N - bi);

		for (int bj = 0; bj < N; bj += S)
		{
			int jSize = (bj + S <= N) ? S : (N - bj);

			for (int bk = 0; bk < N; bk += S)
			{
				int kSize = (bk + S <= N) ? S : (N - bk);

				for (int ii = 0; ii < iSize; ii++)
					for (int kk = 0; kk < kSize; kk++)
						blockA[ii * S + kk] = A[IDX(bi + ii, bk + kk, N)];

				for (int jj = 0; jj < jSize; jj++)
					for (int kk = 0; kk < kSize; kk++)
						blockBT[jj * S + kk] = B[IDX(bk + kk, bj + jj, N)];

				for (int ii = 0; ii < iSize; ii++)
				{
					const double* rowA = &blockA[ii * S];
					for (int jj = 0; jj < jSize; jj++)
					{
						const double* rowBT = &blockBT[jj * S];
						C[IDX(bi + ii, bj + jj, N)] +=
							dotProductUnrolled(rowA, rowBT, kSize, blockUnrollM);
					}
				}
			}
		}
	}

	delete[] blockA;
	delete[] blockBT;
}