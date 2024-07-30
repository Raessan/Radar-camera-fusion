cmake_minimum_required(VERSION 3.16)
project(inference_tensorrt)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 14)

# Including the "release" feature optimizes the performance in production
set(CMAKE_BUILD_TYPE "Release")

# Add the dependency for Eigen3 and OpenCV
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories for Eigen3 and OpenCV
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

# To include the "include" folder to the path, and OpenCV
include_directories(include)

# To compile the library of TensorRT
add_subdirectory(tensorrt_lib/tensorrt_cpp/tensorrt_cpp_lib)
add_subdirectory(tensorrt_lib/tensorrt_cpp/nn_handler_lib)

# To compile the main program
add_executable(inference_tensorrt src/main.cpp)
target_link_libraries(inference_tensorrt PRIVATE nn_handler_lib ${OpenCV_LIBS})