find_package(Eigen3 REQUIRED)

find_package(Pangolin REQUIRED)

add_executable(coord_transform coordinateTransform.cpp)
target_link_libraries(coord_transform Eigen3::Eigen)

add_executable(plot_trajectory plotTrajectory.cpp)
target_link_libraries(plot_trajectory Eigen3::Eigen ${Pangolin_LIBRARIES})