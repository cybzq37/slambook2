cmake_minimum_required(VERSION 3.5)
project(dense_monocular)

set(CMAKE_BUILD_TYPE "Release")

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

############### dependencies ######################
# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        Sophus::Sophus
)

add_executable(dense_mapping dense_mapping.cpp)
target_link_libraries(dense_mapping ${THIRD_PARTY_LIBS})