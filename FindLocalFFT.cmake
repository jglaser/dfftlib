find_package(MKL)

option(ENABLE_HOST "CPU FFT support" ON)
if (MKL_LIBRARIES)
    set(LOCAL_FFT_LIB MKL)
    set(LOCAL_FFT_LIBRARIES "${MKL_LIBRARIES}")
    include_directories(${MKL_INCLUDE_DIR})
endif()

if (NOT LOCAL_FFT_LIB)
    set(ENABLE_HOST OFF)
    message(STATUS "No CPU FFT library found, disabling host interface")
endif()

