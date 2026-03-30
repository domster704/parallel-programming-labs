#include <intrin.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>

struct CpuData {
    unsigned int eax;
    unsigned int ebx;
    unsigned int ecx;
    unsigned int edx;
};

// Основная функция cpuid
CpuData get_cpuid(unsigned int leaf, unsigned int subleaf = 0) {
    int data[4];
    __cpuidex(data, leaf, subleaf);

    CpuData r;
    r.eax = data[0];
    r.ebx = data[1];
    r.ecx = data[2];
    r.edx = data[3];
    return r;
}

bool has_bit(unsigned int value, int bit) {
    return (value & (1u << bit)) != 0;
}

// Красивый вывод для флага поддерживаемости функции
void print_feature(const std::string& name, bool ok) {
    std::cout << "  " << std::setw(15) << std::left << name
        << (ok ? "supported" : "not supported") << "\n";
}

std::string cache_type_name(unsigned int type) {
    if (type == 1) return "Data cache";
    if (type == 2) return "Instruction cache";
    if (type == 3) return "Unified cache";
    return "Unknown";
}

int main() {
    std::cout << "===== CPU Information via CPUID =====\n\n";

    // Производитель
    CpuData leaf0 = get_cpuid(0);
    unsigned int maxBasic = leaf0.eax;

    char vendor[13];
    memcpy(vendor + 0, &leaf0.ebx, 4);
    memcpy(vendor + 4, &leaf0.edx, 4);
    memcpy(vendor + 8, &leaf0.ecx, 4);
    vendor[12] = '\0';

    std::cout << "Vendor: " << vendor << "\n";
    std::cout << "Maximum basic leaf: 0x" << std::hex << maxBasic << std::dec << "\n\n";

    // Максимальный leaf
    CpuData ext0 = get_cpuid(0x80000000);
    unsigned int maxExt = ext0.eax;

    std::cout << "Maximum extended leaf: 0x" << std::hex << maxExt << std::dec << "\n\n";

    // Название процессора
    if (maxExt >= 0x80000004) {
        char brand[49];
        memset(brand, 0, sizeof(brand));

        CpuData b1 = get_cpuid(0x80000002);
        CpuData b2 = get_cpuid(0x80000003);
        CpuData b3 = get_cpuid(0x80000004);

        memcpy(brand + 0, &b1.eax, 4);
        memcpy(brand + 4, &b1.ebx, 4);
        memcpy(brand + 8, &b1.ecx, 4);
        memcpy(brand + 12, &b1.edx, 4);

        memcpy(brand + 16, &b2.eax, 4);
        memcpy(brand + 20, &b2.ebx, 4);
        memcpy(brand + 24, &b2.ecx, 4);
        memcpy(brand + 28, &b2.edx, 4);

        memcpy(brand + 32, &b3.eax, 4);
        memcpy(brand + 36, &b3.ebx, 4);
        memcpy(brand + 40, &b3.ecx, 4);
        memcpy(brand + 44, &b3.edx, 4);

        std::cout << "CPU brand string: " << brand << "\n\n";
    }

    // Leaf 1
    if (maxBasic >= 1) {
        CpuData leaf1 = get_cpuid(1);

        unsigned int stepping = (leaf1.eax >> 0) & 0xF;
        unsigned int model = (leaf1.eax >> 4) & 0xF;
        unsigned int family = (leaf1.eax >> 8) & 0xF;
        unsigned int procType = (leaf1.eax >> 12) & 0x3;
        unsigned int extModel = (leaf1.eax >> 16) & 0xF;
        unsigned int extFamily = (leaf1.eax >> 20) & 0xFF;

        unsigned int displayFamily = family;
        unsigned int displayModel = model;

        if (family == 0xF)
            displayFamily += extFamily;

        if (family == 0x6 || family == 0xF)
            displayModel += (extModel << 4);

        unsigned int logicalCpu = (leaf1.ebx >> 16) & 0xFF;
        unsigned int apicId = (leaf1.ebx >> 24) & 0xFF;

        std::cout << "===== CPUID(1) =====\n";
        std::cout << "Stepping ID: " << stepping << "\n";
        std::cout << "Model: " << model << "\n";
        std::cout << "Family: " << family << "\n";
        std::cout << "Processor Type: " << procType << "\n";
        std::cout << "Extended Model: " << extModel << "\n";
        std::cout << "Extended Family: " << extFamily << "\n";
        std::cout << "Display Model: " << displayModel << "\n";
        std::cout << "Display Family: " << displayFamily << "\n";
        std::cout << "Logical processors: " << logicalCpu << "\n";
        std::cout << "Local APIC ID: " << apicId << "\n\n";

        std::cout << "Features from EDX:\n";
        print_feature("FPU", has_bit(leaf1.edx, 0));
        print_feature("TSC", has_bit(leaf1.edx, 4));
        print_feature("MMX", has_bit(leaf1.edx, 23));
        print_feature("SSE", has_bit(leaf1.edx, 25));
        print_feature("SSE2", has_bit(leaf1.edx, 26));
        print_feature("HTT", has_bit(leaf1.edx, 28));
        std::cout << "\n";

        std::cout << "Features from ECX:\n";
        print_feature("SSE3", has_bit(leaf1.ecx, 0));
        print_feature("SSSE3", has_bit(leaf1.ecx, 9));
        print_feature("FMA3", has_bit(leaf1.ecx, 12));
        print_feature("SSE4.1", has_bit(leaf1.ecx, 19));
        print_feature("SSE4.2", has_bit(leaf1.ecx, 20));
        print_feature("AVX", has_bit(leaf1.ecx, 28));
        std::cout << "\n";
    }

    // Leaf 7
    if (maxBasic >= 7) {
        CpuData leaf7 = get_cpuid(7, 0);

        std::cout << "===== CPUID(7, 0) =====\n";
        print_feature("AVX2", has_bit(leaf7.ebx, 5));
        print_feature("RTM", has_bit(leaf7.ebx, 11));
        print_feature("AVX512F", has_bit(leaf7.ebx, 16));
        print_feature("SHA", has_bit(leaf7.ebx, 29));
        print_feature("GFNI", has_bit(leaf7.ecx, 8));
        print_feature("AMX-BF16", has_bit(leaf7.edx, 22));
        print_feature("AMX-TILE", has_bit(leaf7.edx, 24));
        print_feature("AMX-INT8", has_bit(leaf7.edx, 25));
        std::cout << "\n";

        if (leaf7.eax >= 1) {
            CpuData leaf71 = get_cpuid(7, 1);

            std::cout << "===== CPUID(7, 1) =====\n";
            print_feature("AVX10", has_bit(leaf71.edx, 19));
            print_feature("AMX-COMPLEX", has_bit(leaf71.edx, 8));
            std::cout << "\n";
        }
    }

    // Частоты
    if (maxBasic >= 0x16) {
        CpuData leaf16 = get_cpuid(0x16);

        unsigned int baseFreq = leaf16.eax & 0xFFFF;
        unsigned int maxFreq = leaf16.ebx & 0xFFFF;
        unsigned int busFreq = leaf16.ecx & 0xFFFF;

        std::cout << "===== CPUID(16h) =====\n";
        if (baseFreq == 0 && maxFreq == 0 && busFreq == 0) {
            std::cout << "Frequency information is not reported by this CPU.\n\n";
        }
        else {
            std::cout << "Base frequency: " << baseFreq << " MHz\n";
            std::cout << "Maximum frequency: " << maxFreq << " MHz\n";
            std::cout << "Bus frequency: " << busFreq << " MHz\n\n";
        }
    }

	// Расширенные функции
    if (maxExt >= 0x80000001) {
        CpuData ex1 = get_cpuid(0x80000001);

        std::cout << "===== CPUID(80000001h) =====\n";
        print_feature("SSE4a", has_bit(ex1.ecx, 6));
        print_feature("FMA4", has_bit(ex1.ecx, 16));
        print_feature("3DNowExt", has_bit(ex1.edx, 30));
        print_feature("3DNow", has_bit(ex1.edx, 31));
        std::cout << "\n";
    }

    // Кэш
    bool isAMD = (std::string(vendor) == "AuthenticAMD");
    unsigned int cacheLeaf = isAMD ? 0x8000001D : 0x4;

    bool cacheSupported =
        (!isAMD && maxBasic >= 0x4) ||
        (isAMD && maxExt >= 0x8000001D);

    if (cacheSupported) {
        std::cout << "===== Cache Information =====\n";

        for (unsigned int i = 0; ; i++) {
            CpuData c = get_cpuid(cacheLeaf, i);

            unsigned int cacheType = c.eax & 0x1F;
            if (cacheType == 0) break;

            unsigned int level = (c.eax >> 5) & 0x7;
            bool fullAssoc = has_bit(c.eax, 9);
            unsigned int threads = ((c.eax >> 14) & 0xFFF) + 1;

            unsigned int lineSize = (c.ebx & 0xFFF) + 1;
            unsigned int partitions = ((c.ebx >> 12) & 0x3FF) + 1;
            unsigned int ways = ((c.ebx >> 22) & 0x3FF) + 1;
            unsigned int sets = c.ecx + 1;

            unsigned long long cacheSize =
                (unsigned long long)lineSize * partitions * ways * sets;

            std::cout << "Cache #" << i << "\n";
            std::cout << "Type: " << cache_type_name(cacheType) << "\n";
            std::cout << "Level: L" << level << "\n";
            std::cout << "Fully associative: " << (fullAssoc ? "yes" : "no") << "\n";
            std::cout << "Threads/cores sharing cache: " << threads << "\n";
            std::cout << "Line size: " << lineSize << " bytes\n";
            std::cout << "Partitions: " << partitions << "\n";
            std::cout << "Ways: " << ways << "\n";
            std::cout << "Sets: " << sets << "\n";

            if (cacheSize >= 1024 * 1024)
                std::cout << "Cache size: " << cacheSize / (1024 * 1024) << " MB\n\n";
            else
                std::cout << "Cache size: " << cacheSize / 1024 << " KB\n\n";
        }
    }

    std::cout << "===== End of program =====\n";
    return 0;
}