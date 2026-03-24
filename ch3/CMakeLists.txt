cmake_minimum_required(VERSION 3.5)
project(ch3)

set(CMAKE_CXX_STANDARD 11)

# 加入各子工程
add_subdirectory(useEigen)
add_subdirectory(useGeometry)
add_subdirectory(visualizeGeometry)
add_subdirectory(examples)