###############################################################################
#
#  Copyright (c) 2013-2024, Lawrence Livermore National Security, LLC
#  and other libROM project developers. See the top-level COPYRIGHT
#  file for details.
#
#  SPDX-License-Identifier: (Apache-2.0 OR MIT)
#
###############################################################################
set(examples
    poisson_global_rom
    poisson_local_rom_greedy
    dg_advection_global_rom
    dg_advection_local_rom_matrix_interp
    mixed_nonlinear_diffusion
    nonlinear_elasticity_global_rom
    linear_elasticity_global_rom
    maxwell_global_rom
    de_parametric_maxwell_greedy
    maxwell_local_rom_greedy
    elliptic_eigenproblem_global_rom)

foreach(name IN LISTS examples)
   add_executable(${name} ${name}.cpp)
   target_link_libraries(${name} PRIVATE ROM ${CAROM_LIBS})
   target_include_directories(${name} PRIVATE ${CAROM_INCS})
   target_compile_features(${name} PRIVATE cxx_std_17)
endforeach()
