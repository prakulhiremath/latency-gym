cmake_minimum_required(VERSION 3.15...3.27)
project(latency_gym VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native -mtune=native)
    if(NOT APPLE)
        add_compile_options(-flto)
    endif()
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_latency_gym src/bindings.cpp)

target_include_directories(_latency_gym PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(_latency_gym PRIVATE pybind11::module)

set_target_properties(_latency_gym PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/latency_gym"
)

install(
    TARGETS _latency_gym
    DESTINATION latency_gym
)
