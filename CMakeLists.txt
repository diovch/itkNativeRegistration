
cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
project(itkMotionCorrection)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(itkMotionCorrection main.cpp ImageIO.h)

target_link_libraries(itkMotionCorrection ${ITK_LIBRARIES})