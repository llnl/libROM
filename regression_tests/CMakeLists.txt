###############################################################################
#
#  Copyright (c) 2013-2024, Lawrence Livermore National Security, LLC
#  and other libROM project developers. See the top-level COPYRIGHT
#  file for details.
#
#  SPDX-License-Identifier: (Apache-2.0 OR MIT)
#
###############################################################################

set(regression_test_names
        basisComparator
        checkError
        computeSpeedup
        solutionComparator
        fileComparator)

foreach(name IN LISTS regression_test_names)
   add_executable(${name} ${name}.cpp)
   target_link_libraries(${name} PRIVATE ROM ${CAROM_LIBS})
   target_include_directories(${name} PRIVATE ${CAROM_INCS})
   target_compile_features(${name} PRIVATE cxx_std_17)
endforeach()


