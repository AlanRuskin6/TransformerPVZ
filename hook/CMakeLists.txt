cmake_minimum_required(VERSION 3.15)
project(pvz_hook)

# 必须是32位
set(CMAKE_GENERATOR_PLATFORM Win32)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC UTF-8 编码支持
if(MSVC)
    add_compile_options(/utf-8)
endif()

# 源文件
set(SOURCES
    src/dllmain.cpp
    src/game.cpp
    src/bridge.cpp
    src/state.cpp
)

# 创建DLL
add_library(pvz_hook SHARED ${SOURCES})

# 链接Winsock2
target_link_libraries(pvz_hook ws2_32)

# Release优化
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(pvz_hook PRIVATE /O2)
endif()

# 输出目录
set_target_properties(pvz_hook PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
)
