################################################################################
# Variance-reduction demonstration (symmetrization ON vs OFF).
#
# NOT part of the DCA++ test suite. Parked under .claude/symmetry_project/dev/ per the variance-demo
# plan. Wired into a build only when DCA_WITH_VARIANCE_DEMO=ON, via a guarded add_subdirectory in the
# top-level CMakeLists.txt (an uncommitted local edit). Relies on DCA_LIBS / DCA_INCLUDE_DIRS being
# defined by the parent scope, so it must be added AFTER those are set.
################################################################################

set(VARIANCE_DEMO_INCLUDES
  ${DCA_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR})

# Square (single-band) arms and FeAs (two-band) arms. Same thin body; only the setup header /
# point-group token differs. FeAs is the claim-B model (declared 2 ops, derived 8).
foreach(_tgt ctaux_variance_demo_on ctaux_variance_demo_off
             fe_as_variance_demo_on fe_as_variance_demo_off)
  add_executable(${_tgt} ${CMAKE_CURRENT_SOURCE_DIR}/${_tgt}.cpp)
  target_include_directories(${_tgt} PRIVATE ${VARIANCE_DEMO_INCLUDES})
  target_link_libraries(${_tgt} PUBLIC ${DCA_LIBS})
  if (MPI_FOUND)
    target_link_libraries(${_tgt} PRIVATE ${MPI_C_LIBRARIES})
    target_include_directories(${_tgt} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
  endif()
endforeach()
