cmake_minimum_required(VERSION 3.13)
project(XGBoost)

set(CMAKE_CXX_STANDARD 14)

add_executable(XGBoost main.c split.c split.h tree.h tree.c inputreader.c inputreader.h config.h classifier.c classifier.h sort.c sort.h matrix_helpers.c matrix_helpers.h)
add_executable(bench bench/tsc_x86.h split.c split.h tree.h tree.c inputreader.c inputreader.h config.h classifier.c classifier.h bench/bench_main.c sort.c sort.h matrix_helpers.c matrix_helpers.h)
add_executable(test test/test_main.cpp split.c split.h tree.h test/pseudorandom.cpp test/pseudorandom.h test/test_sort.cpp test/test_splitfinding.cpp test/test_tree.cpp tree.c inputreader.h config.h classifier.c classifier.h test/transformers.cpp test/transformers.h sort.c sort.h matrix_helpers.c matrix_helpers.h)

target_link_libraries(XGBoost m)
target_link_libraries(bench m)
target_link_libraries(test m)
