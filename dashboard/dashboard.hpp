// dashboard.hpp
#pragma once
#include "metrics.hpp"
#include "ansi.hpp"

#include <atomic>
#include <chrono>
#include <thread>
#include <iomanip>

inline void dashboard_loop(const Metrics& m, std::atomic<bool>& running) {
    using namespace std::chrono_literals;

    ansi_hide_cursor();

    while (running.load(std::memory_order_relaxed)) {
        ansi_clear_screen();
        ansi_move_cursor(1, 1);

        // Header
        ansi_bold();
        ansi_cyan();
        std::cout << " BLAS / DL METRICS DASHBOARD\n";
        ansi_reset();
        std::cout << "----------------------------------------\n\n";

        // Training panel
        ansi_bold();
        std::cout << "Training:\n";
        ansi_reset();
        std::cout << "  Time elapsed   : " << std::fixed << std::setprecision(2)
                  << m.train_time_s.load() << " s\n";
        std::cout << "  Loss           : " << m.loss.load() << "\n";
        std::cout << "  Accuracy       : " << m.accuracy.load() << " %\n";
        std::cout << "  Batch size     : " << m.batch_size.load() << "\n\n";

        // Kernel / BLAS panel
        ansi_bold();
        std::cout << "Current kernel (GEMM):\n";
        ansi_reset();
        std::cout << "  Shape          : "
                  << m.m.load() << " x " << m.k.load()
                  << "  *  " << m.k.load() << " x " << m.n.load() << "\n";
        std::cout << "  Throughput     : " << m.gflops.load() << " GFLOP/s\n";

        std::cout << "\n(Press Ctrl+C to quit)\n";

        std::cout.flush();
        std::this_thread::sleep_for(200ms);  // ~5 FPS
    }

    ansi_show_cursor();
}

