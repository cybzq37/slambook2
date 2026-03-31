cmake_minimum_required(VERSION 3.5)
project(ch8)

set(CMAKE_BUILD_TYPE "Release")

#add_definitions("-DENABLE_SSE")
#set(CMAKE_CXX_FLAGS "-std=c++11 ${SSE_FLAGS} -g -O3 -march=native")

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 全局开启 SSE
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    add_compile_options(-msse4)
    add_compile_definitions(ENABLE_SSE)
else()
    message(STATUS "Non-x86 platform: SSE disabled")
endif()

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(g2o REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ${Eigen3_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS}
)

#include_directories(
#        ${OpenCV_INCLUDE_DIRS}
#        ${G2O_INCLUDE_DIRS}
#        ${Sophus_INCLUDE_DIRS}
#        "/usr/include/eigen3/"
#        ${Pangolin_INCLUDE_DIRS}
#)

add_executable(optical_flow optical_flow.cpp)
target_link_libraries(optical_flow ${OpenCV_LIBS} Sophus::Sophus Eigen3::Eigen ${Pangolin_LIBRARIES})

add_executable(direct_method direct_method.cpp)
target_link_libraries(direct_method ${OpenCV_LIBS} Sophus::Sophus Eigen3::Eigen ${Pangolin_LIBRARIES})
