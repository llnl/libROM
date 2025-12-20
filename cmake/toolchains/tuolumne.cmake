set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_Fortran_COMPILER mpif90)

set(HDF5_ROOT $ENV{HDF5_ROOT})
if("${HDF5_ROOT}" STREQUAL "")
  # If no HDF5 module was loaded, load a default version                                                                                                    
  execute_process(COMMAND bash -c "module load cray-hdf5-parallel && echo $HDF5_ROOT"
                  OUTPUT_VARIABLE hdf_path OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(HDF5_ROOT ${hdf_path})
endif()

set(HDF5_NO_FIND_PACKAGE_CONFIG_FILE true)
set(HDF5_PREFER_PARALLEL true)

set(BLA_VENDOR Cray)
set(BLAS_ROOT $ENV{CRAY_LIBSCI_PREFIX_DIR})
set(BLA_PREFER_PKGCONFIG true)

string(TOLOWER $ENV{PE_ENV} LIBSCI_PREFIX)
set(ScaLAPACK_LIBRARY $ENV{CRAY_LIBSCI_PREFIX_DIR}/lib/libsci_${LIBSCI_PREFIX}_mpi.so)
