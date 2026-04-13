cmake_minimum_required(VERSION 3.5)
project(myslam)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 编译器警告标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
# Release 模式优化标志：O3优化、OpenMP并行、pthread支持
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fopenmp -pthread")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

############### dependencies ######################
# Eigen
find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# G2O
find_package(g2o REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

# gtest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# gflags
find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})

# csparse
# find_package(CSparse REQUIRED)
# include_directories(${CSPARSE_INCLUDE_DIR})

# 显式查找 SuiteSparse (g2o_csparse 的底层依赖)
include_directories(/usr/include/suitesparse)
find_library(CXSPARSE_LIB cxsparse
    PATHS /usr/lib/x86_64-linux-gnu
)

# 定义 g2o 相关库及其依赖
# 注意：链接顺序非常重要，底层库（stuff, cxsparse）必须放在后面
SET(G2O_LIBS 
    g2o_solver_csparse       # 核心求解器
    g2o_csparse_extension    # 扩展
    g2o_types_sba            # SBA 相关类型
    g2o_core                 # 核心框架
    g2o_stuff                # 辅助工具
    ${CSPARSE_LIBRARY}       # 底层 csparse
    ${CHOLMOD_LIBRARIES}     # 底层 cholmod
    fmt::fmt
)

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
        ${Pangolin_LIBRARIES} GL GLU GLEW glut
        ${G2O_LIBS}      
        ${GTEST_BOTH_LIBRARIES}
        glog::glog
        ${GFLAGS_LIBRARIES}
        pthread
        ${CSPARSE_LIBRARY}
        )

enable_testing()

############### source and test ######################
include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(app)
