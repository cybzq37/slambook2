cmake_minimum_required(VERSION 3.5)

project(HelloSLAM)

add_library(hello libHelloSLAM.cpp)
add_library(hello_shared SHARED libHelloSLAM.cpp)

add_executable(helloSLAM helloSLAM.cpp)
target_link_libraries(helloSLAM hello_shared)