cmake_minimum_required(VERSION 3.12.4 FATAL_ERROR)
message(STATUS "CMake Version: ${CMAKE_VERSION}")

project(CUDA_GEMM VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

find_package(cuBLAS)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE GEMM_SOURCES "src/*.cu")

add_compile_options("-lineinfo" "-lXptxas-dlcm=ca")

add_library(cudaGemm ${GEMM_SOURCES})

target_compile_features(cudaGemm PUBLIC cxx_std_11)

# tests
add_executable(main tests/main.cu)
target_link_libraries(main
    LINK_PRIVATE
    cudaGemm
    ${CUBLAS_LIBRARIES}
)
target_include_directories(main PRIVATE include)
