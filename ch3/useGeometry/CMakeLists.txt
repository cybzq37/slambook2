cmake_minimum_required(VERSION 3.5)

project(useGeometry)

find_package(Eigen3 REQUIRED)

add_executable(useGeometry useGeometry.cpp)

target_link_libraries(useGeometry Eigen3::Eigen)