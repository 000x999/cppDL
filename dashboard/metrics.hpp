// metrics.hpp
#pragma once
#include <atomic>
#include <cstddef>

struct Metrics {
    std::atomic<double> train_time_s{0.0};
    std::atomic<double> loss{0.0};
    std::atomic<double> accuracy{0.0};      // 0–100
    std::atomic<std::size_t> batch_size{0};

    // Example BLAS / kernel info
    std::atomic<std::size_t> m{0}, n{0}, k{0};   // e.g. GEMM MxK * KxN
    std::atomic<double> gflops{0.0};
};

