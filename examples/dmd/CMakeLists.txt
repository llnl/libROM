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
      dg_advection
      nonlinear_elasticity
      heat_conduction
      heat_conduction_dmdc
      parametric_heat_conduction
      de_parametric_heat_conduction_greedy
      de_dg_advection_greedy
      wave_equation
      dg_euler
      local_dw_csv
      parametric_tw_csv
      parametric_dw_csv
      parametric_dmdc_heat_conduction)

foreach(name IN LISTS examples)
   add_executable(${name} ${name}.cpp)
   target_link_libraries(${name} PRIVATE ROM ${CAROM_LIBS})
   target_include_directories(${name} PRIVATE ${CAROM_INCS})
   target_compile_features(${name} PRIVATE cxx_std_17)
endforeach()
