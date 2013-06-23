add_executable(unit_test unit_test.c kiss_fft.cc kiss_fftnd.cc)
target_link_libraries(unit_test dfft)

if(MPI_COMPILE_FLAGS)
  set_target_properties(unit_test PROPERTIES
  COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(dfft PROPERTIES
        LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

add_test(unit_test "unit_test")
