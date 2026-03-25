cmake_minimum_required(VERSION 3.5)
project(vo1)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")

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

# set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
# list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(g2o REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ${Eigen3_INCLUDE_DIRS}
)

add_executable(orb_cv orb_cv.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBS})

add_executable(orb_self orb_self.cpp)
target_link_libraries(orb_self ${OpenCV_LIBS})

# add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp extra.cpp ) # use this if in OpenCV2 
add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

# # add_executable( triangulation triangulation.cpp extra.cpp) # use this if in opencv2
add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d
        g2o::core g2o::stuff
        ${OpenCV_LIBS})

add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d
        g2o::core g2o::stuff
        ${OpenCV_LIBS})
