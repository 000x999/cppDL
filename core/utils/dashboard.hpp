#pragma once
#include <atomic>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <thread>  
#include <cmath>  

#ifdef _WIN32
  #include <windows.h>
#endif

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

struct LayerRow {
    std::string name;
    std::string type;
    std::string shape;
    int         params_k;  
    double      flops_g;   
    double      ms;        
};

struct TrainDashboardConfig {
    std::size_t num_samples   {0};
    std::size_t input_dim     {0};
    std::size_t output_dim    {0};
    std::size_t batch_size    {0};
    std::size_t num_epochs    {0};
    float       learning_rate {0.0f};
    std::string loss_name     {"MSE"};
};

struct TrainMetrics {
    std::atomic<int>    epoch          {0};
    std::atomic<int>    batch_size     {0};
    std::atomic<double> loss           {0.0}; 
    std::atomic<double> avg_loss       {0.0};  
    std::atomic<double> samples_per_sec{0.0};
    std::atomic<double> train_time_s   {0.0};

    std::atomic<float>  last_out       {0.0f};
    std::atomic<float>  last_target    {0.0f};

    std::atomic<int>    gemm_m         {0};
    std::atomic<int>    gemm_n         {0};
    std::atomic<int>    gemm_k         {0};
    std::atomic<double> gemm_gflops    {0.0};
};


inline void ansi_clear_screen()       { std::cout << "\x1b[2J"; }
inline void ansi_move(int r, int c)   { std::cout << "\x1b[" << r << ";" << c << "H"; }
inline void ansi_hide_cursor()        { std::cout << "\x1b[?25l"; }
inline void ansi_show_cursor()        { std::cout << "\x1b[?25h"; }
inline void ansi_reset()              { std::cout << "\x1b[0m"; }
inline void ansi_bold()               { std::cout << "\x1b[1m"; }
inline void ansi_dim()                { std::cout << "\x1b[2m"; }

inline void fg256(int n)              { std::cout << "\x1b[38;5;" << n << "m"; }
inline void bg256(int n)              { std::cout << "\x1b[48;5;" << n << "m"; }


inline void draw_loss_sparkline(int row, int col, int width,
                                const std::vector<float>& hist)
{
    static const char levels[] = " .:-=+*#%@"; // 10 levels

    float lo = std::numeric_limits<float>::infinity();
    float hi = 0.0f;
    for (float v : hist) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    if (!std::isfinite(lo) || hist.empty()) {
        ansi_move(row, col);
        ansi_dim(); std::cout << "(no data)"; ansi_reset();
        return;
    }
    float range = (hi - lo);
    if (range <= 0.0f) range = 1.0f;

    ansi_move(row, col);
    int n = std::min<int>(width, static_cast<int>(hist.size()));
    int start_idx = static_cast<int>(hist.size()) - n;

    for (int i = 0; i < n; ++i) {
        float v = hist[start_idx + i];
        float norm = (v - lo) / range;     // 0..1
        norm = std::clamp(norm, 0.0f, 1.0f);
        int idx = static_cast<int>(std::round(norm * 9.0f));
        idx = std::clamp(idx, 0, 9);
        char ch = levels[idx];

        if (norm < 0.5)      fg256(34);
        else if (norm < 0.8) fg256(220);
        else                 fg256(196);

        std::cout << ch;
        ansi_reset();
    }
}

inline void draw_box(int top, int left, int width, int height, const std::string& title) {
    ansi_move(top, left);
    fg256(31);
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

    for (int row = 1; row <= height - 2; ++row) {
        ansi_move(top + row, left);
        fg256(31);
        std::cout << "│";
        ansi_move(top + row, left + width - 1);
        std::cout << "│";
    }
    ansi_reset();

    ansi_move(top + height - 1, left);
    fg256(31);
    std::cout << "└";
    for (int i = 0; i < inner; ++i) std::cout << "─";
    std::cout << "┘";
    ansi_reset();
}

inline void draw_bar(int row, int col, int width, double fraction) {
    fraction = std::clamp(fraction, 0.0, 1.0);
    int filled = static_cast<int>(std::round(fraction * width));

    ansi_move(row, col);
    std::cout << "[";
    for (int i = 0; i < width; ++i) {
        if (i < filled) {
            double x = fraction;
            if (x < 0.5)      fg256(34);  
            else if (x < 0.8) fg256(220); 
            else              fg256(196); 
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

inline void draw_layer_table(int top, int left, int width, int height,
                             const std::vector<LayerRow>& rows,
                             int highlight_idx)
{
    int content_top  = top + 1;
    int content_left = left + 1;
    int inner_w      = width - 2;
    int inner_h      = height - 2;

    ansi_move(content_top, content_left);
    bg256(236); fg256(15);
    std::cout << std::left
              << std::setw(12) << "Layer"
              << std::setw(12) << "Type"
              << std::setw(18) << "Shape"
              << std::setw(10) << "ParamsK"
              << std::setw(10) << "GFLOPs"
              << std::setw(8)  << "ms";
    ansi_reset();

    int max_rows = inner_h - 2;
    if (max_rows < 1) return;

    if (highlight_idx < 0) highlight_idx = 0;
    if (highlight_idx >= (int)rows.size()) highlight_idx = (int)rows.size() - 1;

    int start_row = 0;
    if (highlight_idx >= max_rows) {
        start_row = highlight_idx - max_rows + 1;
        if (start_row < 0) start_row = 0;
    }

    for (int i = 0; i < max_rows; ++i) {
        int idx = start_row + i;
        ansi_move(content_top + 1 + i, content_left);
        if (idx < 0 || idx >= (int)rows.size()) {
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
                  << std::setw(12) << r.name
                  << std::setw(12) << r.type
                  << std::setw(18) << r.shape
                  << std::setw(10) << r.params_k
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.flops_g
                  << std::setw(8)  << std::fixed << std::setprecision(2) << r.ms;
        ansi_reset();
    }
}


inline void dashboard_loop(const TrainMetrics& m,
                           const TrainDashboardConfig& cfg,
                           const std::vector<LayerRow>& layers,
                           std::atomic<bool>& running)
{
    using namespace std::chrono_literals;

    ansi_hide_cursor();
    ansi_clear_screen();

    const int term_width = 120;

    static const char spinner_chars[] = "|/-\\";
    int frame = 0;

    while (running.load(std::memory_order_relaxed)) {
        int    epoch       = m.epoch.load();
        int    bs          = m.batch_size.load();
        double loss        = m.loss.load();
        double avg_loss    = m.avg_loss.load();
        double sps         = m.samples_per_sec.load();
        double t_s         = m.train_time_s.load();
        float  last_out    = m.last_out.load();
        float  last_target = m.last_target.load();

        int    gm = m.gemm_m.load();
        int    gn = m.gemm_n.load();
        int    gk = m.gemm_k.load();
        double gg = m.gemm_gflops.load();

        ansi_move(1, 1);
        fg256(45); ansi_bold();
        std::cout << " cppDL / CRUSHBLAS Training Dashboard ";
        ansi_reset();

        ansi_move(1, 60);
        ansi_dim();
        std::cout << "[engine " << spinner_chars[frame % 4] << "]";
        ansi_reset();

        ansi_move(1, term_width - 20);
        ansi_dim();
        std::cout << "Ctrl+C to exit";
        ansi_reset();


        int train_top  = 3, train_left = 2,  train_w = 70, train_h = 15;
        int gemm_top   = 3, gemm_left  = train_left + train_w + 2, gemm_w  = 46, gemm_h  = 9;
        int layer_top  = train_top + train_h + 1;
        int layer_left = 2, layer_w = term_width - 4, layer_h = 10;

        draw_box(train_top, train_left, train_w, train_h, "Training");
        draw_box(gemm_top,  gemm_left,  gemm_w,  gemm_h,  "GEMM / Kernels");
        draw_box(layer_top, layer_left, layer_w, layer_h, "Layers");

        ansi_move(train_top + 2, train_left + 3);
        fg256(15); ansi_bold(); std::cout << "Epoch"; ansi_reset();
        std::cout << ": " << epoch << " / " << (int)cfg.num_epochs;

        ansi_move(train_top + 3, train_left + 3);
        fg256(15); ansi_bold(); std::cout << "Loss (batch)"; ansi_reset();
        std::cout << ": " << std::fixed << std::setprecision(6) << loss;

        ansi_move(train_top + 4, train_left + 3);
        fg256(15); ansi_bold(); std::cout << "Loss (avg)  "; ansi_reset();
        std::cout << ": " << std::fixed << std::setprecision(6) << avg_loss;

        ansi_move(train_top + 5, train_left + 3);
        fg256(15); ansi_bold(); std::cout << "Samples/s"; ansi_reset();
        std::cout << ": " << std::fixed << std::setprecision(1) << sps;

        ansi_move(train_top + 2, train_left + 35);
        fg256(15); ansi_bold(); std::cout << "LR"; ansi_reset();
        std::cout << ": ";
        fg256(33);
        std::cout << std::fixed << std::setprecision(8)  
          << cfg.learning_rate;

        ansi_move(train_top + 3, train_left + 35);
        fg256(15); ansi_bold(); std::cout << "Batch"; ansi_reset();
        std::cout << ": " << bs << " / " << cfg.batch_size;

        ansi_move(train_top + 4, train_left + 35);
        fg256(15); ansi_bold(); std::cout << "Dims"; ansi_reset();
        std::cout << ": " << cfg.input_dim << " → " << cfg.output_dim;

        ansi_move(train_top + 5, train_left + 35);
        fg256(15); ansi_bold(); std::cout << "Loss type"; ansi_reset();
        std::cout << ": " << cfg.loss_name;

        ansi_move(train_top + 7, train_left + 3);
        fg256(15); ansi_bold(); std::cout << "Out[0] / Target[0]"; ansi_reset();
        std::cout << ": " << std::fixed << std::setprecision(4)
                  << last_out << " / " << last_target;

        ansi_move(train_top + 7, train_left + 40);
        fg256(15); ansi_bold(); std::cout << "Elapsed"; ansi_reset();
        std::cout << ": " << std::fixed << std::setprecision(2)
                  << t_s << " s";

        double prog = (cfg.num_epochs > 0)
                      ? (double(epoch) / (double)cfg.num_epochs)
                      : 0.0;
        ansi_move(train_top + 8, train_left + 3);
        ansi_dim(); std::cout << "Epoch progress:"; ansi_reset();
        draw_bar(train_top + 8, train_left + 20, 40, prog);

        ansi_move(gemm_top + 2, gemm_left + 3);
        fg256(15); ansi_bold(); std::cout << "Last GEMM"; ansi_reset();
        std::cout << ": " << gm << " x " << gk
                  << " · " << gk << " x " << gn;

        ansi_move(gemm_top + 3, gemm_left + 3);
        fg256(15); ansi_bold(); std::cout << "Throughput"; ansi_reset();
        std::cout << ": " << std::fixed << std::setprecision(1)
                  << gg << " GFLOP/s";

        ansi_move(gemm_top + 5, gemm_left + 3);
        ansi_dim();
        ansi_move(gemm_top + 6, gemm_left + 3);
        ansi_reset();

        int highlight_idx = (frame / 8) % std::max<int>(1, (int)layers.size());
        draw_layer_table(layer_top, layer_left, layer_w, layer_h,
                         layers, highlight_idx);

        ++frame;
        std::cout.flush();
        std::this_thread::sleep_for(100ms);
    }

    ansi_show_cursor();
}

