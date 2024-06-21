cmake_minimum_required(VERSION 3.16)
project(nn_handler_lib)


# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 14)

# Including the "release" feature optimizes the performance in production
set(CMAKE_BUILD_TYPE "Release")

# # To compile the library of TensorRT
# add_subdirectory(../tensorrt_cpp_api)

# find dependencies
find_package(Threads REQUIRED)

# To include the "include" folder to the path, and OpenCV
# include_directories(include)

# Configuration of the composition libraries
add_library(nn_handler_lib SHARED
  src/nn_handler.cpp
)

target_include_directories(nn_handler_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nn_handler_lib PUBLIC tensorrt_cpp_api)