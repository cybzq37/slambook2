cmake_minimum_required(VERSION 3.5)

project(bundle_adjustment)
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

# LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# 查找必要的依赖库
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)
find_package(g2o REQUIRED)

# 显式查找 SuiteSparse (g2o_csparse 的底层依赖)
include_directories(/usr/include/suitesparse)
find_library(CXSPARSE_LIB cxsparse
    PATHS /usr/lib/x86_64-linux-gnu
)

# 包含路径
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    ${CHOLMOD_INCLUDE_DIR}
    ${CSPARSE_INCLUDE_DIR}
    /usr/include/suitesparse
)

# 库文件查找路径（确保能搜到 /usr/local/lib 下的 g2o）
link_directories(/usr/local/lib /usr/lib/x86_64-linux-gnu)

# 定义 g2o 相关库及其依赖
# 注意：链接顺序非常重要，底层库（stuff, cxsparse）必须放在后面
SET(G2O_LIBS 
    g2o_solver_csparse       # 核心求解器
    g2o_csparse_extension    # 扩展
    g2o_core                 # 核心框架
    g2o_stuff                # 辅助工具
    ${CSPARSE_LIBRARY}       # 底层 csparse
    ${CHOLMOD_LIBRARIES}     # 底层 cholmod
    fmt::fmt
)

# 编译公共库
add_library(bal_common common.cpp)

# 1. 编译 Ceres 版本
add_executable(bundle_adjustment_ceres bundle_adjustment_ceres.cpp)
target_link_libraries(bundle_adjustment_ceres ${CERES_LIBRARIES} bal_common)

# 2. 编译 G2O 版本
add_executable(bundle_adjustment_g2o bundle_adjustment_g2o.cpp)
target_link_libraries(bundle_adjustment_g2o 
    ${G2O_LIBS}
    bal_common
)
