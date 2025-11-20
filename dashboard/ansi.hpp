// ansi.hpp
#pragma once
#include <iostream>

inline void ansi_clear_screen() {
    std::cout << "\x1b[2J";
}

inline void ansi_move_cursor(int row, int col) {
    std::cout << "\x1b[" << row << ";" << col << "H";
}

inline void ansi_hide_cursor() {
    std::cout << "\x1b[?25l";
}

inline void ansi_show_cursor() {
    std::cout << "\x1b[?25h";
}

inline void ansi_reset() {
    std::cout << "\x1b[0m";
}

inline void ansi_bold() {
    std::cout << "\x1b[1m";
}

inline void ansi_green() {
    std::cout << "\x1b[32m";
}

inline void ansi_cyan() {
    std::cout << "\x1b[36m";
}

