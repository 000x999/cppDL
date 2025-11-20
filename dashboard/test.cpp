// dashboard_htop_layers.cpp
#include <iostream>
#include <iomanip>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <cmath>
#include <algorithm>

#ifdef _WIN32
  #include <windows.h>
#endif

// ================== Windows console init (UTF-8 + ANSI) ==================

inline void init_console_utf8() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD mode = 0;
        if (GetConsoleMode(hOut, &mode)) {
            mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, mode);
        }
    }
#endif
}

// ================== Metrics ==================

struct Metrics {
    std::atomic<double> train_time_s{0.0};
    std::atomic<double> loss{0.0};
    std::atomic<double> accuracy{0.0};    // %
    std::atomic<int>    epoch{0};
    std::atomic<int>    batch_size{0};

    // GEMM / kernel info
    std::atomic<int> m{0}, n{0}, k{0};
    std::atomic<double> gflops{0.0};

    // utilization
    std::atomic<double> cpu_like_util{0.0};   // 0–1
};

struct LayerRow {
    const char* name;
    const char* type;
    const char* shape;
    int         params_k;   // params in K
    double      flops_g;    // FLOPs in G
    double      ms;
};

// ================== ANSI helpers ==================

inline void ansi_clear_screen()       { std::cout << "\x1b[2J"; }
inline void ansi_move(int r, int c)   { std::cout << "\x1b[" << r << ";" << c << "H"; }
inline void ansi_hide_cursor()        { std::cout << "\x1b[?25l"; }
inline void ansi_show_cursor()        { std::cout << "\x1b[?25h"; }
inline void ansi_reset()              { std::cout << "\x1b[0m"; }
inline void ansi_bold()               { std::cout << "\x1b[1m"; }
inline void ansi_dim()                { std::cout << "\x1b[2m"; }

inline void fg256(int n)              { std::cout << "\x1b[38;5;" << n << "m"; }
inline void bg256(int n)              { std::cout << "\x1b[48;5;" << n << "m"; }

// ================== UI helpers ==================

void draw_box(int top, int left, int width, int height, const std::string& title) {
    // Top border
    ansi_move(top, left);
    fg256(31); // teal-ish
    std::cout << "┌";
    int inner = width - 2;
    int title_len = static_cast<int>(title.size());
    int title_start = std::max(0, (inner - title_len) / 2);
    for (int i = 0; i < inner; ++i) {
        if (i == title_start && title_len > 0) {
            ansi_bold(); fg256(45);
            std::cout << title;
            ansi_reset(); fg256(31);
            i += title_len - 1;
        } else {
            std::cout << "─";
        }
    }
    std::cout << "┐";
    ansi_reset();

    // Sides
    for (int row = 1; row <= height - 2; ++row) {
        ansi_move(top + row, left);
        fg256(31);
        std::cout << "│";
        ansi_move(top + row, left + width - 1);
        std::cout << "│";
    }
    ansi_reset();

    // Bottom
    ansi_move(top + height - 1, left);
    fg256(31);
    std::cout << "└";
    for (int i = 0; i < inner; ++i) std::cout << "─";
    std::cout << "┘";
    ansi_reset();
}

void draw_bar(int row, int col, int width, double fraction) {
    fraction = std::clamp(fraction, 0.0, 1.0);
    int filled = static_cast<int>(std::round(fraction * width));

    ansi_move(row, col);
    std::cout << "[";
    for (int i = 0; i < width; ++i) {
        if (i < filled) {
            double x = fraction;
            if (x < 0.5)      fg256(34);   // green
            else if (x < 0.8) fg256(220);  // yellow
            else              fg256(196);  // red
            std::cout << "█";
            ansi_reset();
        } else {
            ansi_dim();
            std::cout << " ";
            ansi_reset();
        }
    }
    std::cout << "]";
}

void draw_history(int row, int col, int width, const double* hist) {
    static const char levels[] = " .:-=+*#%@"; // 10 levels
    ansi_move(row, col);
    for (int i = 0; i < width; ++i) {
        double v = std::clamp(hist[i], 0.0, 1.0);
        int idx = static_cast<int>(std::round(v * 9.0));
        idx = std::clamp(idx, 0, 9);
        char ch = levels[idx];

        if (v < 0.5)      fg256(34);
        else if (v < 0.8) fg256(220);
        else              fg256(196);

        std::cout << ch;
        ansi_reset();
    }
}

void draw_layer_table(int top, int left, int width, int height,
                      const LayerRow* rows, int n_rows, int highlight_idx)
{
    int content_top  = top + 1;
    int content_left = left + 1;
    int inner_w      = width - 2;
    int inner_h      = height - 2;

    // header
    ansi_move(content_top, content_left);
    bg256(236); fg256(15);
    std::cout << std::left
              << std::setw(11) << "Layer"
              << std::setw(9)  << "Type"
              << std::setw(16) << "Shape"
              << std::setw(10) << "ParamsK"
              << std::setw(10) << "GFLOPs"
              << std::setw(8)  << "ms";
    ansi_reset();

    int max_rows = inner_h - 2;
    int start_row = 0;
    if (highlight_idx >= max_rows) {
        start_row = highlight_idx - max_rows + 1;
        if (start_row < 0) start_row = 0;
    }

    for (int i = 0; i < max_rows; ++i) {
        int idx = start_row + i;
        ansi_move(content_top + 1 + i, content_left);
        if (idx >= n_rows) {
            std::cout << std::string(inner_w, ' ');
            continue;
        }
        const LayerRow& r = rows[idx];

        bool selected = (idx == highlight_idx);
        if (selected) {
            bg256(24); fg256(255);
        } else if (idx % 2 == 0) {
            bg256(234); fg256(250);
        } else {
            bg256(232); fg256(250);
        }

        std::cout << std::left
                  << std::setw(11) << r.name
                  << std::setw(9)  << r.type
                  << std::setw(16) << r.shape
                  << std::setw(10) << r.params_k
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.flops_g
                  << std::setw(8)  << std::fixed << std::setprecision(2) << r.ms;
        ansi_reset();
    }
}

// ================== Dashboard render loop ==================

void dashboard_loop(const Metrics& m,
                    const LayerRow* layers, int n_layers,
                    std::atomic<bool>& running)
{
    using namespace std::chrono_literals;
    ansi_hide_cursor();
    ansi_clear_screen();

    const int term_width = 120;   // assume ~120 columns; widen your terminal if needed

    static const char spinner_chars[] = "|/-\\";
    int frame = 0;

    static constexpr int HIST_W = 30;
    double util_hist[HIST_W]{};

    while (running.load(std::memory_order_relaxed)) {
        double util = m.cpu_like_util.load();

        for (int i = 0; i < HIST_W - 1; ++i) util_hist[i] = util_hist[i + 1];
        util_hist[HIST_W - 1] = util;

        // header
        ansi_move(1, 1);
        fg256(45); ansi_bold();
        std::cout << " cppDL / CRUSHBLAS dashboard ";
        ansi_reset();

        ansi_move(1, 42);
        ansi_dim();
        std::cout << "[engine ";
        fg256(220);
        std::cout << spinner_chars[frame % 4];
        ansi_reset(); ansi_dim();
        std::cout << "]";
        ansi_reset();

        ansi_move(1, term_width - 26);
        ansi_dim();
        std::cout << "Ctrl+C to exit      ";
        ansi_reset();

        // ---- layout (fixed, content-friendly) ----
        // top row: Training (left), Utilization (right), same height
        int train_top  = 3, train_left = 2,  train_w = 70, train_h = 9;
        int util_top   = 3, util_left  = train_left + train_w + 2, util_w  = 44, util_h  = 15;

        // second row: Kernel panel under Training
        int kern_top   = train_top + train_h + 1;
        int kern_left  = 2, kern_w = 70, kern_h = 8;

        // bottom row: Layers across full width
        int layer_top  = kern_top + kern_h + 1;
        int layer_left = 2, layer_w = term_width - 4, layer_h = 9; // wide enough for table

        draw_box(train_top,  train_left,  train_w,  train_h,  "Training");
        draw_box(util_top,   util_left,   util_w,   util_h,   "Utilization");
        draw_box(kern_top,   kern_left,   kern_w,   kern_h,   "Kernel / GEMM");
        draw_box(layer_top,  layer_left,  layer_w,  layer_h,  "Layers");

        // ---- training panel ----
        ansi_move(train_top + 2, train_left + 3);
        fg256(15); ansi_bold(); std::cout << "Epoch"; ansi_reset();
        std::cout << ": " << m.epoch.load();

        ansi_move(train_top + 2, train_left + 30);
        fg256(15); ansi_bold(); std::cout << "Batch"; ansi_reset();
        std::cout << ": " << m.batch_size.load();

        ansi_move(train_top + 3, train_left + 3);
        fg256(15); ansi_bold(); std::cout << "Time"; ansi_reset();
        std::cout << ": ";
        fg256(220);
        std::cout << std::fixed << std::setprecision(2)
                  << m.train_time_s.load() << " s";
        ansi_reset();

        ansi_move(train_top + 4, train_left + 3);
        fg256(15); ansi_bold(); std::cout << "Loss"; ansi_reset();
        std::cout << ": ";
        fg256(207);
        std::cout << std::setprecision(5) << m.loss.load();
        ansi_reset();

        ansi_move(train_top + 5, train_left + 3);
        fg256(15); ansi_bold(); std::cout << "Acc "; ansi_reset();
        std::cout << ": ";
        fg256(82);
        std::cout << std::setprecision(2) << m.accuracy.load() << " %";
        ansi_reset();

        ansi_move(train_top + 7, train_left + 3);
        ansi_dim(); std::cout << "Train progress:"; ansi_reset();
        double acc_frac = m.accuracy.load() / 100.0;
        draw_bar(train_top + 7, train_left + 20, 40, acc_frac);

        // ---- util panel ----
        ansi_move(util_top + 2, util_left + 3);
        fg256(15); ansi_bold();
        std::cout << "Engine Utilization";
        ansi_reset();

        ansi_move(util_top + 4, util_left + 3);
        std::cout << "Util: ";
        if (util < 0.5)      fg256(34);
        else if (util < 0.8) fg256(220);
        else                 fg256(196);
        std::cout << std::setw(5) << std::setprecision(1) << util * 100.0 << " %";
        ansi_reset();

        draw_bar(util_top + 5, util_left + 3, util_w - 6, util);

        double mem_util = 0.3 + 0.4 * std::sin(util * 6.2831);
        mem_util = std::clamp(mem_util, 0.0, 1.0);

        ansi_move(util_top + 7, util_left + 3);
        std::cout << "Mem : ";
        if (mem_util < 0.5)      fg256(34);
        else if (mem_util < 0.8) fg256(220);
        else                     fg256(196);
        std::cout << std::setw(5) << std::setprecision(1) << mem_util * 100.0 << " %";
        ansi_reset();

        draw_bar(util_top + 8, util_left + 3, util_w - 6, mem_util);

        // history line right under bars
        ansi_move(util_top + 9, util_left + 3);
        ansi_dim(); std::cout << "Hist:"; ansi_reset();
        draw_history(util_top + 9, util_left + 10, HIST_W, util_hist);

        // ---- kernel panel ----
        ansi_move(kern_top + 2, kern_left + 3);
        fg256(15); ansi_bold(); std::cout << "GEMM"; ansi_reset();
        std::cout << "  ";
        fg256(80);
        std::cout << m.m.load() << " x " << m.k.load()
                  << " * " << m.k.load() << " x " << m.n.load();
        ansi_reset();

        ansi_move(kern_top + 3, kern_left + 3);
        fg256(15); ansi_bold(); std::cout << "Throughput"; ansi_reset();
        std::cout << ": ";
        fg256(45);
        std::cout << std::fixed << std::setprecision(1)
                  << m.gflops.load() << " GFLOP/s";
        ansi_reset();

        ansi_move(kern_top + 5, kern_left + 3);
        ansi_dim();
        std::cout << "Use this panel for per-kernel timings, tile sizes,";
        ansi_move(kern_top + 6, kern_left + 3);
        std::cout << "L2 hit rates, etc.";
        ansi_reset();

        // ---- layers panel (wide, bottom) ----
        int highlight_idx = (frame / 5) % std::max(1, n_layers);
        draw_layer_table(layer_top, layer_left, layer_w, layer_h,
                         layers, n_layers, highlight_idx);

        ++frame;
        std::cout.flush();
        std::this_thread::sleep_for(100ms);
    }

    ansi_show_cursor();
}

// ================== Fake metrics producer ==================

void simulate_metrics(Metrics& m, std::atomic<bool>& running) {
    using namespace std::chrono_literals;
    auto start = std::chrono::steady_clock::now();
    int epoch = 0;

    while (running.load(std::memory_order_relaxed)) {
        auto now = std::chrono::steady_clock::now();
        m.train_time_s = std::chrono::duration<double>(now - start).count();

        m.epoch      = epoch;
        m.batch_size = 128 + (epoch % 4) * 128;
        m.loss       = 1.0 / (1.0 + epoch * 0.25);
        m.accuracy   = std::min(100.0, epoch * 3.5);

        m.m = 2048; m.n = 2048; m.k = 2048;
        m.gflops = 3200.0 + 500.0 * std::sin(m.train_time_s.load() * 0.7);

        double t = m.train_time_s.load();
        m.cpu_like_util = 0.5 + 0.5 * std::sin(t * 1.1);

        epoch++;
        std::this_thread::sleep_for(200ms);
    }
}

// ================== main ==================

int main() {
    init_console_utf8();

    LayerRow layers[] = {
        {"conv1",   "Conv2d", "64x112x112",   9408,  14.7, 0.35},
        {"relu1",   "ReLU",   "64x112x112",      0,   0.4, 0.05},
        {"conv2",   "Conv2d", "128x56x56",   73856,  29.3, 0.55},
        {"bn2",     "BN",     "128x56x56",    256,   0.2, 0.03},
        {"fc1",     "Linear", "1024",       524288,  1.0, 0.20},
        {"fc2",     "Linear", "10",          10240,  0.1, 0.02}
    };
    int n_layers = static_cast<int>(std::size(layers));

    Metrics metrics;
    std::atomic<bool> running{true};

    std::thread producer(simulate_metrics, std::ref(metrics), std::ref(running));
    dashboard_loop(metrics, layers, n_layers, running);

    running = false;
    producer.join();
    return 0;
}

