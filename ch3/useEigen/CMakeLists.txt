cmake_minimum_required(VERSION 3.5)

project(useEigen)

find_package(Eigen3 REQUIRED)

add_executable(useEigen eigenMatrix.cpp)

target_link_libraries(useEigen Eigen3::Eigen)
