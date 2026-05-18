###############################################################################
#
#  Copyright (c) 2013-2024, Lawrence Livermore National Security, LLC
#  and other libROM project developers. See the top-level COPYRIGHT
#  file for details.
#
#  SPDX-License-Identifier: (Apache-2.0 OR MIT)
#
###############################################################################

set(ROM_TEST_RANKS 3 CACHE STRING "Number of MPI ranks to use for parallel tests")

set(unit_test_names
    smoke_test
    test_include
    uneven_dist
    weak_scaling
    random_test
    smoke_static
    load_samples)

foreach(name IN LISTS unit_test_names)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE
                          ROM ${MPI_C_LINK_FLAGS} ${MPI_C_LIBRARIES} MPI::MPI_C
                          ${MPI_FORTRAN_LINK_FLAGS} ${MPI_FORTRAN_LIBRARIES}
                          MPI::MPI_Fortran)
    target_include_directories(${name} PRIVATE
                               ${CMAKE_CURRENT_SOURCE_DIR} ${MPI_C_INCLUDE_DIRS})
    target_compile_features(${name} PRIVATE cxx_std_11)
endforeach(name)

set(unit_test_stems
    Vector
    Matrix
    HDFDatabase
    DEIM
    DMD
    GNAT
    QDEIM
    QR
    S_OPT
    StaticSVD
    RandomizedSVD
    IncrementalSVD
    IncrementalSVDBrand
    GreedyCustomSampler
    NNLS
    basis_conversion
    PCHIPInterpolator
    interpolation_gradients)

foreach(stem IN LISTS unit_test_stems)
    add_executable(test_${stem} test_${stem}.cpp)
    target_link_libraries(test_${stem} PRIVATE
                          ROM ${MPI_C_LINK_FLAGS} ${MPI_C_LIBRARIES} MPI::MPI_C
                          ${MPI_FORTRAN_LINK_FLAGS} ${MPI_FORTRAN_LIBRARIES}
                          MPI::MPI_Fortran GTest::GTest)
    target_include_directories(test_${stem} PRIVATE
                               ${CMAKE_CURRENT_SOURCE_DIR} ${MPI_C_INCLUDE_DIRS})
    target_compile_features(test_${stem} PRIVATE cxx_std_11)
    target_compile_definitions(test_${stem} PRIVATE CAROM_HAS_GTEST)

    # serial tests
    set(test_command sh -c "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} test_${stem}")
    add_test(NAME test_${stem} COMMAND ${test_command})
endforeach(stem)

# Add parallel tests
set(parallel_tests
    Matrix
    S_OPT
    DMD
    GreedyCustomSampler
    RandomizedSVD
    StaticSVD
    IncrementalSVDBrand
    HDFDatabase
    QR
    NNLS)
set(MPI_TEST_CMD "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ROM_TEST_RANKS} ${MPIEXEC_PREFLAGS}")
foreach(test IN LISTS parallel_tests)
    add_test(NAME test_${test}_mpi COMMAND sh -c "${MPI_TEST_CMD} ./test_${test}")
    set_tests_properties(test_${test}_mpi PROPERTIES
                         TIMEOUT 300
                         PROCESSORS ${ROM_TEST_RANKS}
                         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endforeach()

# Copy testing inputs and scripts to build/tests/
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/baselines/basis_conversion/
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/baselines/basis_conversion/")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/basis_data/
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/basis_data/")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/s_opt_data/
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/s_opt_data/")

# Create test to setup inputs for basis_conversion test
file(COPY ${CMAKE_SOURCE_DIR}/scripts/data/update_basis_format.py
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/")
add_test(NAME basis_conversion_setup
        COMMAND sh -c "cp baselines/basis_conversion/test_basis.000000 . &&
                       cp baselines/basis_conversion/test_basis_snapshot.000000 . &&
                       python3 update_basis_format.py test_basis.000000 &&
                       python3 update_basis_format.py test_basis_snapshot.000000"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_tests_properties(basis_conversion_setup PROPERTIES FIXTURES_SETUP basis_setup)
set_tests_properties(test_basis_conversion PROPERTIES FIXTURES_REQUIRED basis_setup)
