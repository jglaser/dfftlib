include_directories(${MPI_INCLUDE_PATH})
add_library (dfft SHARED dfft_host.c) 
target_link_libraries(dfft ${MPI_LIBRARIES} ${LOCAL_FFT_LIBRARIES})

if(MPI_COMPILE_FLAGS)
  set_target_properties(dfft PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(dfft PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()


install(TARGETS dfft DESTINATION lib)
install(DIRECTORY . DESTINATION "include" FILES_MATCHING PATTERN *.h)
