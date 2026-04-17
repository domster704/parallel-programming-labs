#include <cstdint>
#include <windows.h>
#include <intrin.h>
#include <mmintrin.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

const int N = 1000000;

int64_t dot_cpp(const int8_t* a, const int8_t* b, int n) {
	int64_t sum = 0;
	for (int i = 0; i < n; i++)
		sum += (int16_t)a[i] * (int16_t)b[i];
	return sum;
}

int64_t dot_cpp_unroll2(const int8_t* a, const int8_t* b, int n) {
	int64_t s0 = 0, s1 = 0;
	int i = 0;
	for (; i + 1 < n; i += 2) {
		s0 += (int16_t)a[i] * (int16_t)b[i];
		s1 += (int16_t)a[i + 1] * (int16_t)b[i + 1];
	}
	for (; i < n; i++)
		s0 += (int16_t)a[i] * (int16_t)b[i];
	return s0 + s1;
}

int64_t dot_cpp_unroll4(const int8_t* a, const int8_t* b, int n) {
	int64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
	int i = 0;
	for (; i + 3 < n; i += 4) {
		s0 += (int16_t)a[i] * (int16_t)b[i];
		s1 += (int16_t)a[i + 1] * (int16_t)b[i + 1];
		s2 += (int16_t)a[i + 2] * (int16_t)b[i + 2];
		s3 += (int16_t)a[i + 3] * (int16_t)b[i + 3];
	}
	for (; i < n; i++)
		s0 += (int16_t)a[i] * (int16_t)b[i];
	return s0 + s1 + s2 + s3;
}

int64_t dot_cpp_unroll8(const int8_t* a, const int8_t* b, int n) {
	int64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
	int64_t s4 = 0, s5 = 0, s6 = 0, s7 = 0;
	int i = 0;
	for (; i + 7 < n; i += 8) {
		s0 += (int16_t)a[i] * (int16_t)b[i];
		s1 += (int16_t)a[i + 1] * (int16_t)b[i + 1];
		s2 += (int16_t)a[i + 2] * (int16_t)b[i + 2];
		s3 += (int16_t)a[i + 3] * (int16_t)b[i + 3];
		s4 += (int16_t)a[i + 4] * (int16_t)b[i + 4];
		s5 += (int16_t)a[i + 5] * (int16_t)b[i + 5];
		s6 += (int16_t)a[i + 6] * (int16_t)b[i + 6];
		s7 += (int16_t)a[i + 7] * (int16_t)b[i + 7];
	}
	for (; i < n; i++)
		s0 += (int16_t)a[i] * (int16_t)b[i];
	return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
}

int64_t dot_mmx(const int8_t* a, const int8_t* b, int n) {
	__m64 sum = _mm_setzero_si64();
	__m64 zero = _mm_setzero_si64();

	for (int i = 0; i < n; i += 8) {
		__m64 va = *(__m64*)(a + i);
		__m64 vb = *(__m64*)(b + i);

		__m64 sa = _mm_cmpgt_pi8(zero, va);
		__m64 sb = _mm_cmpgt_pi8(zero, vb);

		__m64 alo = _mm_unpacklo_pi8(va, sa);
		__m64 ahi = _mm_unpackhi_pi8(va, sa);
		__m64 blo = _mm_unpacklo_pi8(vb, sb);
		__m64 bhi = _mm_unpackhi_pi8(vb, sb);

		__m64 p0 = _mm_madd_pi16(alo, blo);
		__m64 p1 = _mm_madd_pi16(ahi, bhi);

		sum = _mm_add_pi32(sum, p0);
		sum = _mm_add_pi32(sum, p1);
	}

	int32_t t[2];
	*(__m64*)t = sum;
	_mm_empty();

	return (int64_t)t[0] + (int64_t)t[1];
}

int64_t dot_mmx_unroll2(const int8_t* a, const int8_t* b, int n) {
	__m64 sum0 = _mm_setzero_si64();
	__m64 sum1 = _mm_setzero_si64();
	__m64 zero = _mm_setzero_si64();

	int i = 0;
	for (; i + 15 < n; i += 16) {
		__m64 va0 = *(__m64*)(a + i);
		__m64 vb0 = *(__m64*)(b + i);
		__m64 sa0 = _mm_cmpgt_pi8(zero, va0);
		__m64 sb0 = _mm_cmpgt_pi8(zero, vb0);
		__m64 alo0 = _mm_unpacklo_pi8(va0, sa0);
		__m64 ahi0 = _mm_unpackhi_pi8(va0, sa0);
		__m64 blo0 = _mm_unpacklo_pi8(vb0, sb0);
		__m64 bhi0 = _mm_unpackhi_pi8(vb0, sb0);
		sum0 = _mm_add_pi32(sum0, _mm_madd_pi16(alo0, blo0));
		sum0 = _mm_add_pi32(sum0, _mm_madd_pi16(ahi0, bhi0));

		__m64 va1 = *(__m64*)(a + i + 8);
		__m64 vb1 = *(__m64*)(b + i + 8);
		__m64 sa1 = _mm_cmpgt_pi8(zero, va1);
		__m64 sb1 = _mm_cmpgt_pi8(zero, vb1);
		__m64 alo1 = _mm_unpacklo_pi8(va1, sa1);
		__m64 ahi1 = _mm_unpackhi_pi8(va1, sa1);
		__m64 blo1 = _mm_unpacklo_pi8(vb1, sb1);
		__m64 bhi1 = _mm_unpackhi_pi8(vb1, sb1);
		sum1 = _mm_add_pi32(sum1, _mm_madd_pi16(alo1, blo1));
		sum1 = _mm_add_pi32(sum1, _mm_madd_pi16(ahi1, bhi1));
	}

	sum0 = _mm_add_pi32(sum0, sum1);

	int32_t t[2];
	*(__m64*)t = sum0;
	_mm_empty();

	int64_t ans = (int64_t)t[0] + (int64_t)t[1];

	for (; i < n; i++)
		ans += (int16_t)a[i] * (int16_t)b[i];

	return ans;
}

int64_t dot_mmx_unroll4(const int8_t* a, const int8_t* b, int n) {
	__m64 s0 = _mm_setzero_si64();
	__m64 s1 = _mm_setzero_si64();
	__m64 s2 = _mm_setzero_si64();
	__m64 s3 = _mm_setzero_si64();
	__m64 zero = _mm_setzero_si64();

	int i = 0;
	for (; i + 31 < n; i += 32) {
		for (int k = 0; k < 4; k++) {
			__m64 va = *(__m64*)(a + i + 8 * k);
			__m64 vb = *(__m64*)(b + i + 8 * k);
			__m64 sa = _mm_cmpgt_pi8(zero, va);
			__m64 sb = _mm_cmpgt_pi8(zero, vb);
			__m64 alo = _mm_unpacklo_pi8(va, sa);
			__m64 ahi = _mm_unpackhi_pi8(va, sa);
			__m64 blo = _mm_unpacklo_pi8(vb, sb);
			__m64 bhi = _mm_unpackhi_pi8(vb, sb);

			if (k == 0) {
				s0 = _mm_add_pi32(s0, _mm_madd_pi16(alo, blo));
				s0 = _mm_add_pi32(s0, _mm_madd_pi16(ahi, bhi));
			}
			else if (k == 1) {
				s1 = _mm_add_pi32(s1, _mm_madd_pi16(alo, blo));
				s1 = _mm_add_pi32(s1, _mm_madd_pi16(ahi, bhi));
			}
			else if (k == 2) {
				s2 = _mm_add_pi32(s2, _mm_madd_pi16(alo, blo));
				s2 = _mm_add_pi32(s2, _mm_madd_pi16(ahi, bhi));
			}
			else {
				s3 = _mm_add_pi32(s3, _mm_madd_pi16(alo, blo));
				s3 = _mm_add_pi32(s3, _mm_madd_pi16(ahi, bhi));
			}
		}
	}

	s0 = _mm_add_pi32(s0, s1);
	s2 = _mm_add_pi32(s2, s3);
	s0 = _mm_add_pi32(s0, s2);

	int32_t t[2];
	*(__m64*)t = s0;
	_mm_empty();

	int64_t ans = (int64_t)t[0] + (int64_t)t[1];

	for (; i < n; i++)
		ans += (int16_t)a[i] * (int16_t)b[i];

	return ans;
}

int64_t dot_mmx_unroll8(const int8_t* a, const int8_t* b, int n) {
	__m64 s[8];
	__m64 zero = _mm_setzero_si64();

	for (int k = 0; k < 8; k++)
		s[k] = _mm_setzero_si64();

	int i = 0;
	for (; i + 63 < n; i += 64) {
		for (int k = 0; k < 8; k++) {
			__m64 va = *(__m64*)(a + i + 8 * k);
			__m64 vb = *(__m64*)(b + i + 8 * k);

			__m64 sa = _mm_cmpgt_pi8(zero, va);
			__m64 sb = _mm_cmpgt_pi8(zero, vb);

			__m64 alo = _mm_unpacklo_pi8(va, sa);
			__m64 ahi = _mm_unpackhi_pi8(va, sa);
			__m64 blo = _mm_unpacklo_pi8(vb, sb);
			__m64 bhi = _mm_unpackhi_pi8(vb, sb);

			s[k] = _mm_add_pi32(s[k], _mm_madd_pi16(alo, blo));
			s[k] = _mm_add_pi32(s[k], _mm_madd_pi16(ahi, bhi));
		}
	}

	__m64 sum = _mm_setzero_si64();
	for (int k = 0; k < 8; k++)
		sum = _mm_add_pi32(sum, s[k]);

	int32_t t[2];
	*(__m64*)t = sum;
	_mm_empty();

	int64_t ans = (int64_t)t[0] + (int64_t)t[1];

	for (; i < n; i++)
		ans += (int16_t)a[i] * (int16_t)b[i];

	return ans;
}

double measure_us_qpc(
	int64_t(*func)(const int8_t*, const int8_t*, int),
	const int8_t* a, const int8_t* b, int n, int repeats)
{
	LARGE_INTEGER freq, t1, t2;
	QueryPerformanceFrequency(&freq);

	volatile int64_t res = 0;

	QueryPerformanceCounter(&t1);
	for (int i = 0; i < repeats; i++)
		res = func(a, b, n);
	QueryPerformanceCounter(&t2);

	double elapsed_sec = double(t2.QuadPart - t1.QuadPart) / double(freq.QuadPart);
	return (elapsed_sec * 1e6) / repeats; // микросекунды
}

int main() {
	vector<int8_t> a(N), b(N);

	mt19937 gen(123);
	uniform_int_distribution<int> dist(-100, 100);

	for (int i = 0; i < N; i++) {
		a[i] = (int8_t)dist(gen);
		b[i] = (int8_t)dist(gen);
	}

	int64_t r1 = dot_cpp(a.data(), b.data(), N);
	int64_t r2 = dot_cpp_unroll2(a.data(), b.data(), N);
	int64_t r3 = dot_cpp_unroll4(a.data(), b.data(), N);
	int64_t r4 = dot_cpp_unroll8(a.data(), b.data(), N);
	int64_t r5 = dot_mmx(a.data(), b.data(), N);
	int64_t r6 = dot_mmx_unroll2(a.data(), b.data(), N);
	int64_t r7 = dot_mmx_unroll4(a.data(), b.data(), N);
	int64_t r8 = dot_mmx_unroll8(a.data(), b.data(), N);

	cout << "Results:\n";
	cout << "CPP           = " << r1 << '\n';
	cout << "CPP x2        = " << r2 << '\n';
	cout << "CPP x4        = " << r3 << '\n';
	cout << "CPP x8        = " << r4 << '\n';
	cout << "MMX           = " << r5 << '\n';
	cout << "MMX x2        = " << r6 << '\n';
	cout << "MMX x4        = " << r7 << '\n';
	cout << "MMX x8        = " << r8 << '\n';

	if (!(r1 == r2 && r1 == r3 && r1 == r4 && r1 == r5 && r1 == r6 && r1 == r7 &&
		r1 == r8)) {
		cout << "\nMismatch\n";
		return 1;
	}

	int repeats = 25;

	double t1 = measure_us_qpc(dot_cpp, a.data(), b.data(), N, repeats);
	double t2 = measure_us_qpc(dot_cpp_unroll2, a.data(), b.data(), N, repeats);
	double t3 = measure_us_qpc(dot_cpp_unroll4, a.data(), b.data(), N, repeats);
	double t4 = measure_us_qpc(dot_cpp_unroll8, a.data(), b.data(), N, repeats);
	double t5 = measure_us_qpc(dot_mmx, a.data(), b.data(), N, repeats);
	double t6 = measure_us_qpc(dot_mmx_unroll2, a.data(), b.data(), N, repeats);
	double t7 = measure_us_qpc(dot_mmx_unroll4, a.data(), b.data(), N, repeats);
	double t8 = measure_us_qpc(dot_mmx_unroll8, a.data(), b.data(), N, repeats);

	cout << fixed << setprecision(3);

	cout << "\nTime per run:\n";
	cout << "CPP           = " << t1 << " us\n";
	cout << "CPP x2        = " << t2 << " us\n";
	cout << "CPP x4        = " << t3 << " us\n";
	cout << "CPP x8        = " << t4 << " us\n";
	cout << "MMX           = " << t5 << " us\n";
	cout << "MMX x2        = " << t6 << " us\n";
	cout << "MMX x4        = " << t7 << " us\n";
	cout << "MMX x8        = " << t8 << " us\n";

	return 0;
}