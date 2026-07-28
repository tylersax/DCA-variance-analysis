################################################################################
# Standalone top-level project for the symmetrization-variance measurement.
#
# Inverts the usual nesting: this project is the top level and pulls the DCA source in as a
# subdirectory, so DCA itself needs no CMake modification. Point it at a DCA checkout with
# -DDCA_SOURCE_DIR=<path>. (The one DCA edit this project relies on is the local_G_k_w(symmetrize)
# flag in ctaux_cluster_solver.hpp -- an ordinary header edit compiled against, not a build hook.)
#
# add_subdirectory gives DCA its own variable scope, so DCA_LIBS/MPI/include-dirs defined there are
# not visible here. CMake *targets* are global, so we link the target names directly (list mirrored
# from DCA's DCA_LIBS) and re-find MPI/LAPACK in this scope.
################################################################################
cmake_minimum_required(VERSION 3.21)
project(symm_variance LANGUAGES C CXX)

set(DCA_SOURCE_DIR "" CACHE PATH "Path to the DCA source tree")
if(NOT DCA_SOURCE_DIR OR NOT EXISTS "${DCA_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "Set -DDCA_SOURCE_DIR=<path to DCA source checkout>")
endif()

add_subdirectory(${DCA_SOURCE_DIR} ${CMAKE_BINARY_DIR}/dca)

find_package(MPI)
find_package(LAPACK REQUIRED)

# Mirror of DCA's DCA_LIBS (targets are global; the variable is not).
set(SYMM_VARIANCE_LIBS
  git_version modules timer cmake_options ctaux ctint ss_ct_hyb g0_interpolation
  function_transform gaussian_quadrature nfft tetrahedron_mesh random
  parallel_util dca_algorithms json dca_hdf5 enumerations profiling function
  convert_domains cluster_domains quantum_domains time_and_frequency_domains
  signals coarsegraining parallel_mpi_concurrency parallel_no_concurrency
  parallel_stdthread lapack models)

# *_symm_variance: the single-iteration bare-bath drivers (symm_variance_main.inc).
# *_bath_drift:    the N-iteration bath-drift check, ROADMAP 3e (bath_drift_main.inc). Same model
#                  setup and same output schema per iteration; only the driver body differs.
foreach(_tgt square_symm_variance fe_as_symm_variance square_bath_drift fe_as_bath_drift)
  add_executable(${_tgt} ${CMAKE_CURRENT_SOURCE_DIR}/${_tgt}.cpp)
  target_include_directories(${_tgt} PRIVATE
    ${DCA_SOURCE_DIR}
    ${DCA_SOURCE_DIR}/include
    ${DCA_SOURCE_DIR}/src/util
    ${DCA_SOURCE_DIR}/libs/simplex_gm_rule
    ${CMAKE_BINARY_DIR}/include        # DCA generates its config headers against CMAKE_BINARY_DIR
    ${CMAKE_BINARY_DIR}/dca/include
    ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${_tgt} PUBLIC ${SYMM_VARIANCE_LIBS} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
  if (MPI_FOUND)
    target_link_libraries(${_tgt} PRIVATE ${MPI_C_LIBRARIES})
    target_include_directories(${_tgt} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
  endif()
endforeach()
