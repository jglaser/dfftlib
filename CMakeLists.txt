cmake_minimum_required (VERSION 2.6)
project (dfftlib)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}" ${CMAKE_MODULE_PATH})
set (dfftlib_VERSION_MAJOR 0)
set (dfftlib_VERSION_MINOR 1)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(MPI REQUIRED)
include(FindLocalFFT.txt)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OpenMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

configure_file (
  "${PROJECT_SOURCE_DIR}/src/dfft_lib_config.h.in"
  "${PROJECT_BINARY_DIR}/dfft_lib_config.h"
   LOCAL_FFT_LIB
   )

include_directories("${PROJECT_BINARY_DIR}")

include_directories("${PROJECT_SOURCE_DIR}/src")

add_subdirectory(src)

install(FILES "${PROJECT_BINARY_DIR}/dfft_lib_config.h" DESTINATION "include")
