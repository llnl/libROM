###############################################################################
#
#  Copyright (c) 2013-2024, Lawrence Livermore National Security, LLC
#  and other libROM project developers. See the top-level COPYRIGHT
#  file for details.
#
#  SPDX-License-Identifier: (Apache-2.0 OR MIT)
#
###############################################################################

# Require CMake version 3.12 or later to expose modern features of
# CMake for maintainability. With additional work, it is possible to
# rewrite the build system generator for CMake 3.10+, 3.8+, 3.1+,
# 3.0+, or 2.8+, with increasing amounts of work as the minimum
# required version is decreased. Notably, CMake's find modules export
# targets (e.g., FindZLIB.cmake exports the ZLIB::ZLIB IMPORTED
# target) as of version 3.1, and CMake supports using the
# <PackageName>_ROOT variables to set the prefix search path for
# find_package(<PackageName>) as of version 3.12.
cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0074 NEW) # Use <PackageName>_ROOT variables in find_package

# Even though this project is a pure C++ library, the C language must be enabled
# for MPI and HDF5 dependency checking. The Fortran language must be enabled
# for Fortran-C name mangling convention detection.
project(libROM
  VERSION 1.0.0
  DESCRIPTION "Model reduction library emphasizing large-scale parallelism"
  HOMEPAGE_URL "https://github.com/LLNL/libROM"
  LANGUAGES C CXX Fortran)

set(libROM_CMAKE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(libROM_CMAKE_MODULE_PATH ${libROM_CMAKE_PATH}/modules)
list(APPEND CMAKE_MODULE_PATH ${libROM_CMAKE_MODULE_PATH})

option(USE_MFEM "Build libROM with MFEM" OFF)
option(MFEM_USE_GSLIB "Build libROM with MFEM using GSLIB" OFF)
option(BUILD_STATIC "Build libROM as a static library" OFF)
option(ENABLE_EXAMPLES "Build examples and regression tests" ON)
option(ENABLE_DOCS "Build docs using Doxygen" OFF)
option(ENABLE_TESTS "Build unit tests. Requires googletest to be installed" OFF)
set(LIBROM_FLAGS "" CACHE STRING "Additional compiler flags used to compile libROM")

## Set a bunch of variables to generate a configure header
# Enable assertion checking if debug symbols generated
if((CMAKE_BUILD_TYPE STREQUAL "Debug") OR
    (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
  set(DEBUG_CHECK_ASSERTIONS "1")
  set(LIBROM_FLAGS "${LIBROM_FLAGS} -Wall")
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_INIT} ${LIBROM_FLAGS}" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_INIT} ${LIBROM_FLAGS}" CACHE STRING "" FORCE)

if (CMAKE_HOST_APPLE)
    # Fix linker flags for OSX to use classic linker on XCode 15.0+:
    # https://developer.apple.com/documentation/xcode-release-notes/xcode-15-release-notes#Linking
    set(MPI_Fortran_LINK_FLAGS "${MPI_Fortran_LINK_FLAGS} -Wl,-ld_classic")
endif()

set(CAROM_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CAROM_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CAROM_VERSION_PATCHLEVEL "${PROJECT_VERSION_PATCH}")
set(CAROM_PACKAGE ${CMAKE_PROJECT_NAME})
set(CAROM_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
set(CAROM_PACKAGE_STRING "${CMAKE_PROJECT_NAME} ${CMAKE_PROJECT_VERSION}")
set(CAROM_PACKAGE_TARNAME ${CMAKE_PROJECT_NAME})
set(CAROM_PACKAGE_URL ${CMAKE_PROJECT_HOMEPAGE_URL})
set(CAROM_PACKAGE_VERSION ${CMAKE_PROJECT_VERSION})
set(CAROM_VERSION ${CMAKE_PROJECT_VERSION})

include(CheckIncludeFiles)
check_include_files(dlfcn.h CAROM_HAVE_DLFCN_H)
check_include_files(inttypes.h CAROM_HAVE_INTTYPES_H)
check_include_files(memory.h CAROM_HAVE_MEMORY_H)
check_include_files(stdint.h CAROM_HAVE_STDINT_H)
check_include_files(stdlib.h CAROM_HAVE_STDLIB_H)
check_include_files(strings.h CAROM_HAVE_STRINGS_H)
check_include_files(string.h CAROM_HAVE_STRING_H)
check_include_files(sys/stat.h CAROM_HAVE_SYS_STAT_H)
check_include_files(sys/types.h CAROM_HAVE_SYS_TYPES_H)
check_include_files(unistd.h CAROM_HAVE_UNISTD_H)

# Construct an expression consisting of the 24 ANSI C headers
set(stdc_header_list "assert.h;complex.h;ctype.h;errno.h;fenv.h;float.h")
set(stdc_header_list "${stdc_header_list};inttypes.h;iso646.h;limits.h")
set(stdc_header_list "${stdc_header_list};locale.h;math.h;setjmp.h;signal.h")
set(stdc_header_list "${stdc_header_list};stdarg.h;stdbool.h;stdint.h")
set(stdc_header_list "${stdc_header_list};stddef.h;stdio.h;stdlib.h;string.h")
set(stdc_header_list "${stdc_header_list};tgmath.h;time.h;wchar.h;wctype.h")
check_include_files("${stdc_header_list}" CAROM_STDC_HEADERS)

# List minimum version requirements for dependencies where possible to make
# packaging easier later.
find_package(HDF5 1.8.0 REQUIRED)
find_package(BLAS 3.4.0 REQUIRED)
find_package(LAPACK 3.4.0 REQUIRED)

# In FindMPI.cmake, "CXX" refers to "the MPI C API [being] usable from C++"
find_package(MPI 1.2 REQUIRED)

find_package(ZLIB 1.2.3 REQUIRED)

if (ENABLE_TESTS)
    find_package(GTest 1.6.0 REQUIRED)
endif()

# Define variables for use in generating a configure file
set(CAROM_HAVE_BLAS ${BLAS_FOUND})
set(CAROM_HAVE_LAPACK ${LAPACK_FOUND})
set(CAROM_HAVE_HDF5 ${HDF5_FOUND})

if (USE_MFEM)
    # Try to find MFEM installed using cmake
    find_package(mfem NAMES MFEM HINTS "${MFEM_DIR}"
                "${MFEM_DIR}/lib/cmake/mfem" "${CMAKE_SOURCE_DIR}/dependencies/mfem/lib/cmake/mfem" NO_DEFAULT_PATH QUIET)
    if (mfem_FOUND)
       message(STATUS "Found mfem config in: ${mfem_DIR} (version ${MFEM_VERSION})")
    else()
       # Try to find MFEM installed using configure
       find_library(MFEM mfem HINTS "${MFEM_DIR}/lib" "${CMAKE_SOURCE_DIR}/dependencies/mfem")
       find_library(HYPRE HYPRE HINTS "${HYPRE_DIR}/lib" "${CMAKE_SOURCE_DIR}/dependencies/hypre/src/hypre/lib")
       find_library(PARMETIS parmetis HINTS "${PARMETIS_DIR}/lib" "${CMAKE_SOURCE_DIR}/dependencies/parmetis-4.0.3/build/lib/libparmetis")
       find_library(METIS metis HINTS "${METIS_DIR}/lib" "${CMAKE_SOURCE_DIR}/dependencies/parmetis-4.0.3/build/lib/libmetis")
       find_path(MFEM_INCLUDES mfem.hpp HINTS "${MFEM_DIR}/include" "${CMAKE_SOURCE_DIR}/dependencies/mfem")
       find_path(HYPRE_INCLUDES HYPRE.h HINTS "${HYPRE_DIR}/include" "${CMAKE_SOURCE_DIR}/dependencies/hypre/src/hypre/include")
       find_path(PARMETIS_INCLUDES metis.h "${PARMETIS_DIR}/metis/include" "${CMAKE_SOURCE_DIR}/dependencies/parmetis-4.0.3/metis/include")

       if (${MFEM} STREQUAL "MFEM-NOTFOUND")
          error( "MFEM not found" )
       else()
          message(STATUS "MFEM library:" ${MFEM})
          message(STATUS "MFEM includes:" ${MFEM_INCLUDES})
       endif()
    endif()
endif()

set(CAROM_LIBS ${MPI_C_LINK_FLAGS} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} MPI::MPI_C ${MPI_FORTRAN_LINK_FLAGS} ${MPI_FORTRAN_LIBRARIES} MPI::MPI_Fortran)
set(CAROM_INCS ${CMAKE_CURRENT_SOURCE_DIR} ${MPI_C_INCLUDE_DIRS})

add_subdirectory(lib)

if (ENABLE_EXAMPLES)
  add_subdirectory(examples/misc)
  if (USE_MFEM)
    add_subdirectory(examples/prom)
    add_subdirectory(examples/dmd)
    file(COPY examples/data DESTINATION ${CMAKE_BINARY_DIR}/examples)
    file(COPY examples/merlin DESTINATION ${CMAKE_BINARY_DIR}/examples)
    file(COPY examples/dmd/heat_conduction_csv.sh DESTINATION ${CMAKE_BINARY_DIR}/examples/dmd)
    file(COPY examples/dmd/heat_conduction_hdf.sh DESTINATION ${CMAKE_BINARY_DIR}/examples/dmd)
    add_subdirectory(regression_tests)
  endif()
endif(ENABLE_EXAMPLES)

if (ENABLE_TESTS)
    enable_testing()
    add_subdirectory(unit_tests tests)
endif()

# NOTE(goxberry@gmail.com, oxberry1@llnl.gov): This code snippet
# builds the Doxygen documentation, but outputs said documentation to
# ${CMAKE_CURRENT_SOURCE_DIR}/docs)...
if(${ENABLE_DOCS})
  find_package(Doxygen 1.8.5 REQUIRED)
  
  set(doxyfile ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile)

  add_custom_target(
    documentation ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
    )
  add_dependencies(documentation ROM)

  add_custom_target(
    doxygen_tagfile
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/rom.tag
    ${CMAKE_CURRENT_BINARY_DIR}/docs/html/rom.tag)
  add_dependencies(doxygen_tagfile documentation)
endif()
