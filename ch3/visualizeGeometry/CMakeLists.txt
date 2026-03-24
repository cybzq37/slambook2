cmake_minimum_required(VERSION 3.5)
project(visualizeGeometry)

# 添加Eigen头文件
find_package(Eigen3 REQUIRED)

# 添加Pangolin依赖
find_package( Pangolin REQUIRED )
include_directories( ${Pangolin_INCLUDE_DIRS} )

add_executable( visualizeGeometry visualizeGeometry.cpp )
target_link_libraries( visualizeGeometry Eigen3::Eigen ${Pangolin_LIBRARIES} )