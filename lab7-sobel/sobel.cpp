#include <immintrin.h>
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// (3 + 10 + 3) * 255 = 16 * 255 = 4080 = H_v_max = H_h_max
// sqrt((H_h_max)^2 + (H_v_max)^2) = 4080 * sqrt(2)
// k = 255 / (4080 * sqrt(2)) = 255 / (16 * 255 * sqrt(2))
static constexpr float SCHARR_SCALE =
255.0f / (16.0f * 255.0f * 1.41421356237f);

double now_ms() {
    static LARGE_INTEGER freq{};
    static bool initialized = false;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = true;
    }

    LARGE_INTEGER counter{};
    QueryPerformanceCounter(&counter);
    return 1000.0 * static_cast<double>(counter.QuadPart) /
        static_cast<double>(freq.QuadPart);
}

void scharr_scalar(const uint8_t* src, uint8_t* dst, int width, int height) {
    std::memset(dst, 0, static_cast<size_t>(width) * height);

    for (int y = 1; y < height - 1; ++y) {
        const uint8_t* r0 = src + (y - 1) * width;
        const uint8_t* r1 = src + y * width;
        const uint8_t* r2 = src + (y + 1) * width;
        uint8_t* out = dst + y * width;

        for (int x = 1; x < width - 1; ++x) {
            const int A = r0[x - 1];
            const int B = r0[x];
            const int C = r0[x + 1];

            const int D = r1[x - 1];
            const int F = r1[x + 1];

            const int G = r2[x - 1];
            const int H = r2[x];
            const int I = r2[x + 1];

            const int H_h = 3 * A + 10 * B + 3 * C - 3 * G - 10 * H - 3 * I;
            const int H_v = 3 * A + 10 * D + 3 * G - 3 * C - 10 * F - 3 * I;

            const float mag = std::sqrt(static_cast<float>(H_h * H_h + H_v * H_v));

            int value = static_cast<int>(mag * SCHARR_SCALE);
            value = std::clamp(value, 0, 255);
            out[x] = static_cast<uint8_t>(value);
        }
    }
}

void scharr_avx512(const uint8_t* src, uint8_t* dst, int width, int height) {
    std::memset(dst, 0, static_cast<size_t>(width) * height);

    const __m512i c3 = _mm512_set1_epi32(3);
    const __m512i c10 = _mm512_set1_epi32(10);
    const __m512 k = _mm512_set1_ps(SCHARR_SCALE);

    alignas(64) int tmp[16];

    for (int y = 1; y < height - 1; ++y) {
        const uint8_t* r0 = src + (y - 1) * width;
        const uint8_t* r1 = src + y * width;
        const uint8_t* r2 = src + (y + 1) * width;
        uint8_t* out = dst + y * width;

        int x = 1;
        for (; x <= width - 1 - 16; x += 16) {
            const __m128i a8 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(r0 + x - 1));
            const __m128i b8 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(r0 + x));
            const __m128i c8 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(r0 + x + 1));

            const __m128i d8 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(r1 + x - 1));
            const __m128i f8 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(r1 + x + 1));

            const __m128i g8 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(r2 + x - 1));
            const __m128i h8 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(r2 + x));
            const __m128i i8 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(r2 + x + 1));

            const __m512i A = _mm512_cvtepu8_epi32(a8);
            const __m512i B = _mm512_cvtepu8_epi32(b8);
            const __m512i C = _mm512_cvtepu8_epi32(c8);

            const __m512i D = _mm512_cvtepu8_epi32(d8);
            const __m512i F = _mm512_cvtepu8_epi32(f8);

            const __m512i G = _mm512_cvtepu8_epi32(g8);
            const __m512i H = _mm512_cvtepu8_epi32(h8);
            const __m512i I = _mm512_cvtepu8_epi32(i8);

            __m512i H_h = _mm512_setzero_si512();
            H_h = _mm512_add_epi32(H_h, _mm512_mullo_epi32(A, c3));
            H_h = _mm512_add_epi32(H_h, _mm512_mullo_epi32(B, c10));
            H_h = _mm512_add_epi32(H_h, _mm512_mullo_epi32(C, c3));
            H_h = _mm512_sub_epi32(H_h, _mm512_mullo_epi32(G, c3));
            H_h = _mm512_sub_epi32(H_h, _mm512_mullo_epi32(H, c10));
            H_h = _mm512_sub_epi32(H_h, _mm512_mullo_epi32(I, c3));

            __m512i H_v = _mm512_setzero_si512();
            H_v = _mm512_add_epi32(H_v, _mm512_mullo_epi32(A, c3));
            H_v = _mm512_add_epi32(H_v, _mm512_mullo_epi32(D, c10));
            H_v = _mm512_add_epi32(H_v, _mm512_mullo_epi32(G, c3));
            H_v = _mm512_sub_epi32(H_v, _mm512_mullo_epi32(C, c3));
            H_v = _mm512_sub_epi32(H_v, _mm512_mullo_epi32(F, c10));
            H_v = _mm512_sub_epi32(H_v, _mm512_mullo_epi32(I, c3));

            const __m512 Hh_ps = _mm512_cvtepi32_ps(H_h);
            const __m512 Hv_ps = _mm512_cvtepi32_ps(H_v);

            __m512 mag = _mm512_sqrt_ps(_mm512_add_ps(_mm512_mul_ps(Hh_ps, Hh_ps), _mm512_mul_ps(Hv_ps, Hv_ps)));

            mag = _mm512_mul_ps(mag, k);

            const __m512i res32 = _mm512_cvttps_epi32(mag);

            _mm512_store_si512(reinterpret_cast<__m512i*>(tmp), res32);

            for (int i = 0; i < 16; ++i) {
                int v = std::clamp(tmp[i], 0, 255);
                out[x + i] = static_cast<uint8_t>(v);
            }
        }

        for (; x < width - 1; ++x) {
            const int A = r0[x - 1];
            const int B = r0[x];
            const int C = r0[x + 1];

            const int D = r1[x - 1];
            const int F = r1[x + 1];

            const int G = r2[x - 1];
            const int H = r2[x];
            const int I = r2[x + 1];

            const int Hh = 3 * A + 10 * B + 3 * C - 3 * G - 10 * H - 3 * I;
            const int Hv = 3 * A + 10 * D + 3 * G - 3 * C - 10 * F - 3 * I;

            const float mag = std::sqrt(static_cast<float>(Hh * Hh + Hv * Hv));

            int value = static_cast<int>(mag * SCHARR_SCALE);
            value = std::clamp(value, 0, 255);
            out[x] = static_cast<uint8_t>(value);
        }
    }
}

bool compare_images(const std::vector<uint8_t>& a,
    const std::vector<uint8_t>& b, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (a[y * width + x] != b[y * width + x]) {
                std::cout << "Mismatch at (" << x << ", " << y
                    << "): " << static_cast<int>(a[y * width + x]) << " vs "
                    << static_cast<int>(b[y * width + x]) << '\n';
                return false;
            }
        }
    }
    return true;
}

int main() {
    constexpr int width = 4096;
    constexpr int height = 4096;
    constexpr int iterations = 30;

    std::vector<uint8_t> src(static_cast<size_t>(width) * height);
    std::vector<uint8_t> dst_scalar(static_cast<size_t>(width) * height);
    std::vector<uint8_t> dst_simd(static_cast<size_t>(width) * height);

    std::mt19937 rng(123456);
    std::uniform_int_distribution<int> dist(0, 255);

    for (auto& px : src) {
        px = static_cast<uint8_t>(dist(rng));
    }

    scharr_scalar(src.data(), dst_scalar.data(), width, height);
    scharr_avx512(src.data(), dst_simd.data(), width, height);

    const bool ok = compare_images(dst_scalar, dst_simd, width, height);
    std::cout << "Compare: " << (ok ? "OK" : "FAILED") << "\n\n";

    if (!ok) {
        return 1;
    }

    double t1 = now_ms();
    for (int i = 0; i < iterations; ++i) {
        scharr_scalar(src.data(), dst_scalar.data(), width, height);
    }
    double t2 = now_ms();

    double t3 = now_ms();
    for (int i = 0; i < iterations; ++i) {
        scharr_avx512(src.data(), dst_simd.data(), width, height);
    }
    double t4 = now_ms();

    const double scalar_ms = (t2 - t1) / iterations;
    const double simd_ms = (t4 - t3) / iterations;
    const double speedup = scalar_ms / simd_ms;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Image size          : " << width << " x " << height << "\n";
    std::cout << "Iterations          : " << iterations << "\n";
    std::cout << "Scalar avg time     : " << scalar_ms << " ms\n";
    std::cout << "AVX-512 avg time    : " << simd_ms << " ms\n";
    std::cout << "Speedup             : " << speedup << "x\n";

    return 0;
}