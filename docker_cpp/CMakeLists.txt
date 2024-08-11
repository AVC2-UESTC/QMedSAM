cmake_minimum_required(VERSION 3.16) 
project(uestcstd)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

add_subdirectory("argparse")

include_directories("xtl/include")
# add_subdirectory("xtl")

include_directories("xsimd/include")
# add_subdirectory("xsimd")

set(BUILD_SHARED_LIBS OFF)

# set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
# set(XTENSOR_USE_XSIMD ON)
# set(DOWNLOAD_GBENCHMARK OFF)
add_compile_definitions(XTENSOR_USE_XSIMD)
include_directories("xtensor/include")
# add_subdirectory("xtensor")

set(ZLIB_COMPAT ON)
set(ZLIB_ENABLE_TESTS OFF)
set(ZLIBNG_ENABLE_TESTS OFF)
set(WITH_GZFILEOP OFF)
set(WITH_GTEST OFF)
add_subdirectory("zlib-ng" "build_zlib-ng")

# set(HAVE_ZLIB ON)
# add_subdirectory("xtensor-io")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/xtensor-io/include/xtensor-io/xtensor_io_config.hpp.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/xtensor-io/include/xtensor-io/xtensor_io_config.hpp"
)
include_directories("xtensor-io/include")
# link zlibstatic manually

# set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenVINO REQUIRED)

add_executable(uestcsd_inference "main.cpp")
set_target_properties(uestcsd_inference PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
target_link_libraries(uestcsd_inference argparse zlibstatic opencv_core opencv_imgproc openvino::runtime)
target_compile_options(uestcsd_inference PRIVATE -march=skylake -mtune=skylake)
