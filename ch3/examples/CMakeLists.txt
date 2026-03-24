cmake_minimum_required(VERSION 3.22)

project(ch3_examples)

find_package(Eigen3)

add_executable(plot_trajectory plotTrajectory.cpp)
target_link_libraries(plot_trajectory Eigen3::Eigen)