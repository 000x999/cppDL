// main.cpp
#include "metrics.hpp"
#include "dashboard.hpp"

#include <thread>
#include <atomic>
#include <chrono>

void run_training(Metrics& m) {
    using clock = std::chrono::steady_clock;
    auto start = clock::now();

    for (int epoch = 0; epoch < 10; ++epoch) {
        // Example updates – plug in your real values
        m.batch_size = 128;
        m.loss       = 1.0 / (epoch + 1);
        m.accuracy   = epoch * 10.0;  // fake

        // Example GEMM config
        m.m = 1024; m.n = 1024; m.k = 1024;
        m.gflops = 4000.0;           // compute from your timing

        auto now = clock::now();
        m.train_time_s =
            std::chrono::duration<double>(now - start).count();

        // Simulate work
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int main() {
    Metrics metrics;
    std::atomic<bool> running{true};

    std::thread ui([&] { dashboard_loop(metrics, running); });

    run_training(metrics);

    running = false;
    ui.join();
}

