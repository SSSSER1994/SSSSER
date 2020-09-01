cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(traffic_match)
add_compile_options(--std=c++11)

find_package(Eigen3 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include/ ${EIGEN3_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} src/main.cpp src/hm_tracks_objects_match.cpp src/track_object_distance.cpp)

target_link_libraries(${PROJECT_NAME} ${EIGEN3_LIBRARIES})
