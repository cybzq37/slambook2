cmake_minimum_required(VERSION 3.5)

set(CMAKE_BUILD_TYPE Release)

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

# opencv 
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# eigen 
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# pcl 
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

# octomap 
find_package(octomap REQUIRED)
include_directories(${OCTOMAP_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

add_executable(pointcloud_mapping pointcloud_mapping.cpp)
target_link_libraries(pointcloud_mapping ${OpenCV_LIBS} ${PCL_LIBRARIES} Sophus::Sophus)

add_executable(octomap_mapping octomap_mapping.cpp)
target_link_libraries(octomap_mapping ${OpenCV_LIBS} ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES} Sophus::Sophus)

add_executable(surfel_mapping surfel_mapping.cpp)
target_link_libraries(surfel_mapping ${OpenCV_LIBS} ${PCL_LIBRARIES} Sophus::Sophus)
