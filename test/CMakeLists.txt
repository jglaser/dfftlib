if(ENABLE_HOST)
    add_executable(unit_test_host unit_test_host.c kiss_fft.c kiss_fftnd.c)
    target_link_libraries(unit_test_host dfft m)

    include_directories(${MPI_INCLUDE_PATH})
    if(MPI_COMPILE_FLAGS)
      set_target_properties(unit_test_host PROPERTIES
      COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    endif()

    if(MPI_LINK_FLAGS)
    set_target_properties(unit_test_host PROPERTIES
        LINK_FLAGS "${MPI_LINK_FLAGS}")
    endif()
    add_test(unit_test_host "unit_test_host")
endif()

if(ENABLE_CUDA)
    add_executable(unit_test_cuda unit_test_cuda.c kiss_fft.c kiss_fftnd.c)
    target_link_libraries(unit_test_cuda dfft m)

    include_directories(${MPI_INCLUDE_PATH} ${CUDA_TOOLKIT_INCLUDE})
    if(MPI_COMPILE_FLAGS)
      set_target_properties(unit_test_cuda PROPERTIES
      COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    endif()

    if(MPI_LINK_FLAGS)
        set_target_properties(unit_test_cuda PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
    endif()
    add_test(unit_test_cuda "unit_test_cuda")
endif()

