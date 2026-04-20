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

static inline void zero_borders(uint8_t* dst, int width, int height) {
    if (width <= 0 || height <= 0) {
        return;
    }

    std::memset(dst, 0, static_cast<size_t>(width));
    if (height > 1) {
        std::memset(dst + static_cast<size_t>(width) * (height - 1), 0,
            static_cast<size_t>(width));
    }

    for (int y = 1; y < height - 1; ++y) {
        dst[static_cast<size_t>(y) * width] = 0;
        dst[static_cast<size_t>(y) * width + (width - 1)] = 0;
    }
}

void scharr_scalar(const uint8_t* __restrict src, uint8_t* __restrict dst,
    int width, int height) {
    zero_borders(dst, width, height);

    for (int y = 1; y < height - 1; ++y) {
        const uint8_t* r0 = src + static_cast<size_t>(y - 1) * width;
        const uint8_t* r1 = src + static_cast<size_t>(y) * width;
        const uint8_t* r2 = src + static_cast<size_t>(y + 1) * width;
        uint8_t* out = dst + static_cast<size_t>(y) * width;

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

static inline void store_16_u8_from_epi32(uint8_t* dst, __m512i v32) {
    const __m256i lo256 = _mm512_castsi512_si256(v32);
    const __m256i hi256 = _mm512_extracti64x4_epi64(v32, 1);

    const __m128i v0 = _mm256_castsi256_si128(lo256);
    const __m128i v1 = _mm256_extracti128_si256(lo256, 1);
    const __m128i v2 = _mm256_castsi256_si128(hi256);
    const __m128i v3 = _mm256_extracti128_si256(hi256, 1);

    const __m128i p01_16 = _mm_packus_epi32(v0, v1);
    const __m128i p23_16 = _mm_packus_epi32(v2, v3);
    const __m128i p_8 = _mm_packus_epi16(p01_16, p23_16);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), p_8);
}

static inline __m512i scharr_16bit_kernel_epi16(__m512i A, __m512i B, __m512i C,
    __m512i D, __m512i F, __m512i G,
    __m512i H, __m512i I,
    bool horizontal) {
    const __m512i c3 = _mm512_set1_epi16(3);
    const __m512i c10 = _mm512_set1_epi16(10);

    if (horizontal) {
        __m512i t = _mm512_setzero_si512();
        t = _mm512_add_epi16(t, _mm512_mullo_epi16(A, c3));
        t = _mm512_add_epi16(t, _mm512_mullo_epi16(B, c10));
        t = _mm512_add_epi16(t, _mm512_mullo_epi16(C, c3));
        t = _mm512_sub_epi16(t, _mm512_mullo_epi16(G, c3));
        t = _mm512_sub_epi16(t, _mm512_mullo_epi16(H, c10));
        t = _mm512_sub_epi16(t, _mm512_mullo_epi16(I, c3));
        return t;
    }
    else {
        __m512i t = _mm512_setzero_si512();
        t = _mm512_add_epi16(t, _mm512_mullo_epi16(A, c3));
        t = _mm512_add_epi16(t, _mm512_mullo_epi16(D, c10));
        t = _mm512_add_epi16(t, _mm512_mullo_epi16(G, c3));
        t = _mm512_sub_epi16(t, _mm512_mullo_epi16(C, c3));
        t = _mm512_sub_epi16(t, _mm512_mullo_epi16(F, c10));
        t = _mm512_sub_epi16(t, _mm512_mullo_epi16(I, c3));
        return t;
    }
}

static inline __m512i magnitude_16_epi32(__m256i gx16, __m256i gy16) {
    const __m512i gx32 = _mm512_cvtepi16_epi32(gx16);
    const __m512i gy32 = _mm512_cvtepi16_epi32(gy16);

    const __m512 gx_ps = _mm512_cvtepi32_ps(gx32);
    const __m512 gy_ps = _mm512_cvtepi32_ps(gy32);

    __m512 mag =
        _mm512_add_ps(_mm512_mul_ps(gx_ps, gx_ps), _mm512_mul_ps(gy_ps, gy_ps));
    mag = _mm512_sqrt_ps(mag);
    mag = _mm512_mul_ps(mag, _mm512_set1_ps(SCHARR_SCALE));

    return _mm512_cvttps_epi32(mag);
}

void scharr_avx512_fast(const uint8_t* __restrict src, uint8_t* __restrict dst,
    int width, int height) {
    zero_borders(dst, width, height);

    for (int y = 1; y < height - 1; ++y) {
        const uint8_t* r0 = src + static_cast<size_t>(y - 1) * width;
        const uint8_t* r1 = src + static_cast<size_t>(y) * width;
        const uint8_t* r2 = src + static_cast<size_t>(y + 1) * width;
        uint8_t* out = dst + static_cast<size_t>(y) * width;

        int x = 1;

        for (; x <= width - 1 - 32; x += 32) {
            const __m256i a8 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + x - 1));
            const __m256i b8 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + x));
            const __m256i c8 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + x + 1));

            const __m256i d8 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + x - 1));
            const __m256i f8 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + x + 1));

            const __m256i g8 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r2 + x - 1));
            const __m256i h8 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r2 + x));
            const __m256i i8 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r2 + x + 1));

            const __m512i A = _mm512_cvtepu8_epi16(a8);
            const __m512i B = _mm512_cvtepu8_epi16(b8);
            const __m512i C = _mm512_cvtepu8_epi16(c8);

            const __m512i D = _mm512_cvtepu8_epi16(d8);
            const __m512i F = _mm512_cvtepu8_epi16(f8);

            const __m512i G = _mm512_cvtepu8_epi16(g8);
            const __m512i H = _mm512_cvtepu8_epi16(h8);
            const __m512i I = _mm512_cvtepu8_epi16(i8);

            const __m512i gx16_all =
                scharr_16bit_kernel_epi16(A, B, C, D, F, G, H, I, true);
            const __m512i gy16_all =
                scharr_16bit_kernel_epi16(A, B, C, D, F, G, H, I, false);

            const __m256i gx16_lo = _mm512_castsi512_si256(gx16_all);
            const __m256i gx16_hi = _mm512_extracti64x4_epi64(gx16_all, 1);
            const __m256i gy16_lo = _mm512_castsi512_si256(gy16_all);
            const __m256i gy16_hi = _mm512_extracti64x4_epi64(gy16_all, 1);

            const __m512i mag32_lo = magnitude_16_epi32(gx16_lo, gy16_lo);
            const __m512i mag32_hi = magnitude_16_epi32(gx16_hi, gy16_hi);

            store_16_u8_from_epi32(out + x, mag32_lo);
            store_16_u8_from_epi32(out + x + 16, mag32_hi);
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

            const int H_h = 3 * A + 10 * B + 3 * C - 3 * G - 10 * H - 3 * I;
            const int H_v = 3 * A + 10 * D + 3 * G - 3 * C - 10 * F - 3 * I;

            const float mag = std::sqrt(static_cast<float>(H_h * H_h + H_v * H_v));

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
            if (a[static_cast<size_t>(y) * width + x] !=
                b[static_cast<size_t>(y) * width + x]) {
                std::cout << "Mismatch at (" << x << ", " << y << "): "
                    << static_cast<int>(a[static_cast<size_t>(y) * width + x])
                    << " vs "
                    << static_cast<int>(b[static_cast<size_t>(y) * width + x])
                    << '\n';
                return false;
            }
        }
    }
    return true;
}

int main() {
    constexpr int width = 4096;
    constexpr int height = width;
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
    scharr_avx512_fast(src.data(), dst_simd.data(), width, height);

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
        scharr_avx512_fast(src.data(), dst_simd.data(), width, height);
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