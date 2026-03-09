cmake_minimum_required(VERSION 3.15)
project(mandeye_convert LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Eigen 3 — header-only math library used by LasLoader.h
# ---------------------------------------------------------------------------
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found locally — downloading via FetchContent…")
    include(FetchContent)
    FetchContent_Declare(
        eigen3
        URL      https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    )
    set(EIGEN_BUILD_DOC     OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING       OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen3)
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${eigen3_SOURCE_DIR}")
    endif()
endif()

# ---------------------------------------------------------------------------
# LASzip — in-tree, build as static to produce a self-contained EXE
# ---------------------------------------------------------------------------
set(LASZIP_BUILD_STATIC ON CACHE BOOL "Build LASzip as static lib" FORCE)

# Compute target names the same way LASzip's own CMakeLists does
set(_LZ_CORE "laszip")
set(_LZ_API  "laszip_api")
if(WIN32)
    set(_LZ_CORE "laszip3")
    set(_LZ_API  "laszip_api3")
endif()

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/3rd/LASzip
    ${CMAKE_CURRENT_BINARY_DIR}/LASzip_build
)

# ---------------------------------------------------------------------------
# mandeye_convert executable
# ---------------------------------------------------------------------------
add_executable(mandeye_convert
    mandeye_convert.cpp
    ../common/LasLoader.cpp
    ../common/ImuLoader.cpp
)

target_include_directories(mandeye_convert PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}             # for ros1bag_writer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/..          # so #include "common/..." works
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/3rd/LASzip/include
    ${CMAKE_CURRENT_BINARY_DIR}/LASzip_build/include  # generated laszip_api_version.h
)

target_link_libraries(mandeye_convert PRIVATE
    ${_LZ_API}
    ${_LZ_CORE}
    Eigen3::Eigen
)

if(MSVC)
    target_compile_options(mandeye_convert PRIVATE /W3 /MP)
    target_compile_definitions(mandeye_convert PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

# Place the EXE in a predictable location
set_target_properties(mandeye_convert PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

message(STATUS "mandeye_convert: standalone Windows/Linux EXE — no ROS required")
