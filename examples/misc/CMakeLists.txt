set(examples
    combine_samples)

foreach(name IN LISTS examples)
   add_executable(${name} ${name}.cpp)
   target_link_libraries(${name} PRIVATE ROM ${CAROM_LIBS})
   target_include_directories(${name} PRIVATE ${CAROM_INCS})
   target_compile_features(${name} PRIVATE cxx_std_17)
endforeach()
