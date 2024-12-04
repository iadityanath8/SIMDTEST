cmake_minimum_required(VERSION 3.10)


project(vecsimd
    VERSION 1.0
    DESCRIPTION "A SIMD-based vector library for int and double"
    LANGUAGES CXX
)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


include_directories(include)

add_executable(vecsimd src/main.cc)

target_compile_options(vecsimd PRIVATE -mavx2)